import abc
import numpy as np
import tensorflow as tf

from rdcnet.losses.utils import hingify


def relabel_sequential(labels):
    '''Relabel positive (foreground) labels sequentially'''
    fg_mask = labels > 0
    _, seq_labels = tf.unique(tf.boolean_mask(labels, fg_mask))

    return tf.tensor_scatter_nd_update(labels, tf.where(fg_mask),
                                       seq_labels + 1)


class InstanceEmbeddingLossBase(tf.keras.losses.Loss):
    '''Base class for embedding losses.

    '''
    def __init__(self, parallel_iterations=4, *args, **kwargs):
        '''
        '''
        super().__init__(*args, **kwargs)
        self.parallel_iterations = parallel_iterations

    def call(self, y_true, y_pred):
        '''
        '''
        y_true = tf.cast(y_true, tf.int32)

        # remove batch item that have no groundtruth at all
        has_instance_mask = tf.reduce_any(y_true > 0,
                                          axis=tuple(
                                              range(1, len(y_true.shape))))

        def map_to_not_empty():
            y_true_masked = tf.boolean_mask(y_true, has_instance_mask, axis=0)
            y_pred_masked = tf.boolean_mask(y_pred, has_instance_mask, axis=0)

            loss = tf.map_fn(self._unbatched_loss,
                             [y_true_masked, y_pred_masked],
                             tf.float32,
                             parallel_iterations=self.parallel_iterations)

            return tf.reduce_mean(loss)

        return tf.cond(tf.reduce_any(has_instance_mask), map_to_not_empty,
                       lambda: 0.)

    @abc.abstractmethod
    def _unbatched_loss(self, packed):
        '''
        '''
        pass


class SpatialInstanceEmbeddingLossBase(InstanceEmbeddingLossBase):
    '''Base class for losses converting embeddigns to distance to instance center'''
    def __init__(self, eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps

    def _unbatched_soft_jaccard(self, y_true, y_pred):
        '''expects y_true as one-hot and y_pred as probabilities between [0, 1]
    
        '''
        spatial_axis = tuple(range(len(y_true.shape) - 1))
        intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)

        # apply to foreground only
        fg_mask = tf.cast(
            tf.reduce_any(tf.greater_equal(y_true, 0.5),
                          axis=-1,
                          keepdims=True), tf.float32)
        union = tf.reduce_sum(fg_mask * (y_pred + y_true),
                              axis=spatial_axis) - intersection

        jaccard = 1 - (intersection + self.eps) / (union + self.eps)

        return jaccard

    def _unbatched_label_to_hot(self, instance_labels):
        '''
        Generates 1-hot encoding of instance labels.
        
        Notes:
        ignores negative labels and background=0
        '''

        spatial_axis = tuple(range(len(instance_labels.shape) - 1))

        # remove background for one-hot by making it negative.
        instance_labels = instance_labels - 1
        n_classes = tf.maximum(0, tf.reduce_max(instance_labels)) + 1

        hot = tf.one_hot(tf.squeeze(instance_labels, -1), n_classes)

        return hot

    def _unbatched_embedding_center(self, hot, y_pred):
        '''Returns the mean of each embedding under the true instance mask'''

        spatial_axis = tuple(range(len(hot.shape) - 1))

        # mean embedding under the true instance mask
        counts = tf.expand_dims(tf.reduce_sum(hot, axis=spatial_axis), -1)
        y_pred = tf.expand_dims(y_pred, -2)
        centers = tf.reduce_sum((tf.expand_dims(hot, -1) * y_pred),
                                axis=spatial_axis,
                                keepdims=True) / counts

        return centers

    def _unbatched_embeddings_to_center_dist(self, embeddings, centers):
        '''Returns a embedding distance map per center'''

        # add 1hot dimension to embeddings
        embeddings = tf.expand_dims(embeddings, -2)

        return tf.norm(centers - embeddings, axis=-1)

    @abc.abstractmethod
    def _center_dist_to_probs(self, one_hot, center_dist):
        pass

    def _unbatched_loss(self, packed):
        '''
        '''

        y_true, y_pred = packed
        y_true = relabel_sequential(y_true)  # on random patch level
        one_hot = self._unbatched_label_to_hot(y_true)

        centers = self._unbatched_embedding_center(one_hot, y_pred)
        center_dist = self._unbatched_embeddings_to_center_dist(
            y_pred, centers)

        probs = self._center_dist_to_probs(one_hot, center_dist)

        return tf.reduce_mean(self._unbatched_soft_jaccard(one_hot, probs))


class InstanceMeanIoUEmbeddingLoss(SpatialInstanceEmbeddingLossBase):
    '''
    
    Neven, D., Brabandere, B.D., Proesmans, M. and Gool, L.V., 2019. 
    Instance segmentation by jointly optimizing spatial embeddings and 
    clustering bandwidth. In Proceedings of the IEEE Conference on 
    Computer Vision and Pattern Recognition (pp. 8837-8845).
    '''
    def __init__(self, margin, clip_probs=None, *args, **kwargs):
        '''
        
        Args:
            margin: distance from center where instance probability = 0.5
            clip_probs: clips probabilities values if a (low,high) tuple is provided
        '''
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.clip_probs = clip_probs

        assert self.margin > 0.

    def _center_dist_to_probs(self, one_hot, center_dist):
        '''
        Converts embeddings to probability maps by passing their distances 
        from the given centers through a gaussian function:
        
        p(e_i) = exp(-2 * (norm(e_i-center)/sigma)**2)
        
        where: margin = sigma * sqrt(-2 * ln(0.5))
        
        i.e. embeddings further than margin away from a center have a probability < 0.5
            
        Notes:
        
        For more details see
        
        Neven, Davy, et al. "Instance segmentation by jointly optimizing spatial
        embeddings and clustering bandwidth." Proceedings of the IEEE Conference
        on Computer Vision and Pattern Recognition. 2019.
        '''

        sigma = self.margin * (-2 * np.log(0.5))**-0.5
        probs = tf.exp(-0.5 * (center_dist / sigma)**2)

        if self.clip_probs is None:
            return probs
        elif isinstance(self.clip_probs, tuple) and len(self.clip_probs) == 2:
            return tf.clip_by_value(probs, *self.clip_probs)
        else:
            raise ValueError(
                'clip_probs should be None or (low,high) tuple: . got {}'.
                format(self.clip_probs))


class MarginInstanceEmbeddingLoss(SpatialInstanceEmbeddingLossBase):
    '''Same as InstanceMeanIoUEmbeddingLoss except embeddings are 
    converted to a probability map with intra/inter margins "hinges" 
    instead of a gaussian kernel.
    
    Similar to:
    De Brabandere, B., Neven, D. and Van Gool, L., 2017. Semantic 
    instance segmentation with a discriminative loss function. arXiv 
    preprint arXiv:1708.02551.
    
    '''
    def __init__(self, intra_margin, inter_margin, *args, **kwargs):
        '''
        
        Args:
            intra_margin: distance from an embedding to its center below which the loss is zero
            inter_margin: distance from an embedding to other centers above which the loss is zero
        '''
        super().__init__(*args, **kwargs)

        self.intra_margin = intra_margin
        self.inter_margin = inter_margin

        assert self.intra_margin > 0.
        assert self.inter_margin > self.intra_margin

    def _center_dist_to_probs(self, one_hot, center_dist):
        '''Applies f(d) linear transform to distance d so that f(intra_margin)==1 and f(inter_margin)==0'''

        a = 1 / (self.intra_margin - self.inter_margin)
        b = self.inter_margin / (self.inter_margin - self.intra_margin)
        rescaled_center_dist = a * center_dist + b

        return hingify(one_hot, rescaled_center_dist)
