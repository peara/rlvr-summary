"""Data loaders for CNN-DailyMail and other datasets."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class CNNDMLoader:
    """Loader for CNN-DailyMail dataset.
    
    Provides utilities to load CNN-DailyMail dataset from various sources
    including HuggingFace datasets, local files, or custom formats.
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize CNN-DM loader.
        
        Args:
            data_path: Path to local data files (optional)
            split: Dataset split to load ('train', 'validation', 'test')
            cache_dir: Directory to cache downloaded data
            max_samples: Maximum number of samples to load (for development)
        """
        self.data_path = Path(data_path) if data_path else None
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        self._dataset = None
        
    def load_from_huggingface(self) -> Iterator[Dict]:
        """Load CNN-DM dataset from HuggingFace datasets.
        
        Returns:
            Iterator over dataset samples with 'article', 'highlights', 'id' fields
        """
        try:
            # Try to import datasets library
            from datasets import load_dataset
            
            logger.info(f"Loading CNN-DM dataset split '{self.split}' from HuggingFace")
            dataset = load_dataset(
                "cnn_dailymail", 
                "3.0.0", 
                split=self.split,
                cache_dir=self.cache_dir
            )
            
            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))
                
            self._dataset = dataset
            
            for item in dataset:
                yield {
                    "id": item.get("id", ""),
                    "article": item.get("article", ""),
                    "highlights": item.get("highlights", ""),
                    "url": item.get("url", ""),
                }
                
        except ImportError:
            logger.warning("datasets library not available, falling back to local loading")
            yield from self.load_from_local()
            
    def load_from_local(self) -> Iterator[Dict]:
        """Load CNN-DM dataset from local files.
        
        Expected format: JSONL files with 'article', 'highlights', 'id' fields
        
        Returns:
            Iterator over dataset samples
        """
        if not self.data_path:
            raise ValueError("data_path must be specified for local loading")
            
        file_path = self.data_path / f"{self.split}.jsonl"
        
        if not file_path.exists():
            # Try alternative naming conventions
            alternative_paths = [
                self.data_path / f"cnn_dm_{self.split}.jsonl",
                self.data_path / f"cnn_dailymail_{self.split}.jsonl",
                self.data_path / f"{self.split}.json",
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    file_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find dataset file for split '{self.split}' in {self.data_path}")
                
        logger.info(f"Loading CNN-DM dataset from {file_path}")
        
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if self.max_samples and count >= self.max_samples:
                        break
                        
                    item = json.loads(line.strip())
                    
                    # Ensure required fields are present
                    yield {
                        "id": item.get("id", f"local_{line_num}"),
                        "article": item.get("article", item.get("text", "")),
                        "highlights": item.get("highlights", item.get("summary", "")),
                        "url": item.get("url", ""),
                    }
                    count += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                    
    def load(self) -> Iterator[Dict]:
        """Load CNN-DM dataset using the best available method.
        
        Returns:
            Iterator over dataset samples
        """
        if self.data_path and self.data_path.exists():
            return self.load_from_local()
        else:
            return self.load_from_huggingface()
            
    def get_dataset_info(self) -> Dict:
        """Get information about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics and metadata
        """
        if self._dataset is None:
            # Load dataset to get info
            list(self.load())
            
        info = {
            "split": self.split,
            "data_path": str(self.data_path) if self.data_path else "huggingface",
            "max_samples": self.max_samples,
        }
        
        if hasattr(self._dataset, "__len__"):
            info["num_samples"] = len(self._dataset)
            
        return info


class CustomDataLoader:
    """Generic data loader for custom dataset formats."""
    
    def __init__(self, data_path: Union[str, Path]):
        """Initialize custom data loader.
        
        Args:
            data_path: Path to data file or directory
        """
        self.data_path = Path(data_path)
        
    def load_jsonl(self, max_samples: Optional[int] = None) -> Iterator[Dict]:
        """Load data from JSONL format.
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            Iterator over data samples
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if max_samples and count >= max_samples:
                        break
                        
                    item = json.loads(line.strip())
                    yield item
                    count += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                    
    def load_json(self, max_samples: Optional[int] = None) -> Iterator[Dict]:
        """Load data from JSON format.
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            Iterator over data samples
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            if max_samples:
                data = data[:max_samples]
            for item in data:
                yield item
        else:
            yield data