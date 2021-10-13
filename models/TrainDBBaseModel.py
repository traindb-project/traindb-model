import abc

class TrainDBModel(abc.ABC):
  """Base class for all the ``TrainDB`` models."""

  def train(self, real_data, table_metadata): 
    pass

  def save(self, output_path):
    pass

  def load(self, input_path):
    pass


class TrainDBSynopsisModel(TrainDBModel, abc.ABC):
  """Base class for all the ``TrainDB`` synopsis generation models."""

  def synopsis(self, row_count):
    pass
