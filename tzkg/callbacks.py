from typing import List

import torch
import numpy as np


class Callback(object):
    """Callback function for training model

    Args:

    """

    def __init__(self) -> None:
        pass
    
    def on_train_begin(self):
        pass
    
    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        return True

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        return True
    
    def set_model(self, model):
        self.model = model

class ModelCheckpoint(Callback):
    def __init__(self,
                filepath,
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                save_freq='epoch',
                options=None,
                initial_value_threshold=None,
                **kwargs):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.mode = mode
        
        self._options = options
        self.best = initial_value_threshold

        # init monitor_op
        if mode == 'min':
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf

    def on_train_begin(self, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch, batch=None, logs=logs)
    
    def _save_model(self, epoch, batch, logs=None):
        """ 保存模型
         
         Args:
            epoch: 当前迭代的epoch
            batch: 
            logs: 训练日志
        """
        filepath = self._get_file_path(epoch, batch, logs)

        # save model
        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if self.monitor_op(current, self.best):
                    self.best = current
                    torch.save(self.model.state_dict(), filepath)
                    # self.model.save(filepath, overwrite=True, options=self._options)
            else:
                torch.save(self.model.state_dict(), filepath)
                # self.model.save(filepath, overwrite=True, options=self._options)
        except Exception as e:
            raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}`,`{batch:02d}`
            # and `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            if batch is None or 'batch' in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(epoch=epoch + 1, batch=batch + 1, **logs)
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f'Reason: {e}')
        
        return file_path

class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, monitor="val_loss", min_delta=0, patience=0, mode="auto", verbose=0, restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        # init monitor
        if mode == "max":
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf  # if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_begin(self, epoch):
        return super().on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, logs=None):
        """epoch训练结束之后，根据metric判断是否需要终止。

        Args:
            epoch (_type_): _description_

        Returns:
            _type_: _description_
        """
        current = self.get_monitor_value(logs)
        if current is None:
            return

        # 更新状态
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            # if self.restore_best_weights:
            #     self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            # if self.baseline is None or self._is_improvement(current, self.baseline):

            self.wait = 0

        # 判断是否需要提前终止
        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

            # if self.restore_best_weights and self.best_weights is not None:
            #     if self.verbose > 0:
            #     io_utils.print_msg(
            #         'Restoring model weights from the end of the best epoch: '
            #         f'{self.best_epoch + 1}.')
            #     self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)

        # if monitor_value is None:
        # logging.warning('Early stopping conditioned on metric `%s` '
        #                 'which is not available. Available metrics are: %s',
        #                 self.monitor, ','.join(list(logs.keys())))
        return monitor_value


class CallbackList(object):
    def __init__(self, callbacks: List[Callback], model=None):
        self.callbacks = callbacks
        
        self.set_model(model)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
