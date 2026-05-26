from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from razordl.core.base import logging
from razordl.core.engine.on_policy_single_model.config import Config
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data_config
        self.dataset_processor = config.data_config.dataset_processor


TrainCollator = DataCollatorWithPadding
