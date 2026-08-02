import logging
import logging.handlers
import multiprocessing


def _setup_worker_logging(queue: multiprocessing.Queue) -> logging.Logger:
    """
    Route all log records from this worker process through the shared queue.
    Returns a module-level logger ready for use.
    """
    root          = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(logging.DEBUG)
    return logging.getLogger(__name__)


def init_worker(cfg, queue):
    global GLOBAL_CFG
    GLOBAL_CFG = cfg
    _setup_worker_logging(queue)