"""image_dataset dataset."""

from email.mime import image
import tensorflow_datasets as tfds

# TODO(image_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(image_dataset): BibTeX citation
_CITATION = """
"""


class ImageDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for image_dataset dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Set dir path to images
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(image_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(180, 180, 3)),
            'label': tfds.features.ClassLabel(names=['above', 'below']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(image_dataset): Downloads the data and defines the splits
    image_path = dl_manager.manual_dir / 'images/Above'
    path = dl_manager.extract(image_path)

    # TODO(image_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
        'test': self._generate_examples(path / 'test_images'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(image_dataset): Yields (key, example) tuples from the dataset
    for img_path in path.glob('*.jpeg'):
      yield img_path.name, {
          'image': img_path,
          'label': 'above' if img_path.name.startswith('Above') else 'below',
      }
