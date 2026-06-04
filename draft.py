import logging
import logging.config
LOGGING = {
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
      "standard": {
          "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
      },
  },
  "handlers": {
      "console": {
          "class": "logging.StreamHandler",
          "formatter": "standard",
          "level": "INFO",
      },
      "train_file": {
          "class": "logging.FileHandler",
          "filename": "train.log",
          "formatter": "standard",
          "level": "INFO",
      },
      "eval_file": {
          "class": "logging.FileHandler",
          "filename": "eval.log",
          "formatter": "standard",
          "level": "INFO",
      },
  },
  "loggers": {
      "train": {
          "handlers": ["console", "train_file"],
          "level": "INFO",
          "propagate": False,
      },
      "eval": {
          "handlers": ["console", "eval_file"],
          "level": "INFO",
          "propagate": False,
      },
  },
}

logging.config.dictConfig(LOGGING)

train_log = logging.getLogger("train")
eval_log = logging.getLogger("eval")

train_log.info("开始训练")
eval_log.info("开始评估")
