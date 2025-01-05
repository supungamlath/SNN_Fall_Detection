from clearml import Logger


def report_metrics(
    metric_type,
    metric_dict,
    epoch,
):
    logger = Logger.current_logger()
    for key, value in metric_dict.items():
        logger.report_scalar(metric_type, key, iteration=epoch, value=value)
