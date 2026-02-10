from .rec_metric import RecMetric


class RecGTCMetric(object):

    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 stream=False,
                 with_ratio=False,
                 max_len=25,
                 max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.gtc_metric = RecMetric(main_indicator=main_indicator,
                                    is_filter=is_filter,
                                    ignore_space=ignore_space,
                                    stream=stream,
                                    with_ratio=with_ratio,
                                    max_len=max_len,
                                    max_ratio=max_ratio)
        self.ctc_metric = RecMetric(main_indicator=main_indicator,
                                    is_filter=is_filter,
                                    ignore_space=ignore_space,
                                    stream=stream,
                                    with_ratio=with_ratio,
                                    max_len=max_len,
                                    max_ratio=max_ratio)

    def __call__(self,
                 pred_label,
                 batch=None,
                 training=False,
                 *args,
                 **kwargs):
        # infer_gtc=False면 ctc만 반환됨 (list가 아님)
        if not isinstance(pred_label, list):
            ctc_metric = self.ctc_metric(pred_label, batch, training=training)
            return ctc_metric
        ctc_metric = self.ctc_metric(pred_label[1], batch, training=training)
        gtc_metric = self.gtc_metric(pred_label[0], batch, training=training)
        ctc_metric['gtc_acc'] = gtc_metric['acc']
        ctc_metric['gtc_norm_edit_dis'] = gtc_metric['norm_edit_dis']
        return ctc_metric

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        ctc_metric = self.ctc_metric.get_metric()
        gtc_metric = self.gtc_metric.get_metric()
        # gtc_metric에 값이 있을 때만 추가
        if gtc_metric.get('acc', 0) > 0 or gtc_metric.get('norm_edit_dis', 0) > 0:
            ctc_metric['gtc_acc'] = gtc_metric['acc']
            ctc_metric['gtc_norm_edit_dis'] = gtc_metric['norm_edit_dis']
        return ctc_metric
