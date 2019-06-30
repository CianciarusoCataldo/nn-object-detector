
import keras
from ..utils.coco_eval import evaluate_coco


class CocoEval(keras.callbacks.Callback):
    def __init__(self, generator, threshold=0.05):
        self.generator = generator
        self.threshold = threshold

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        evaluate_coco(self.generator, self.model, self.threshold)
