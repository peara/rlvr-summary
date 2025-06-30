"""Batch processing utilities for data pipeline operations."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import json

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processing utilities for data pipeline operations.
    
    Provides efficient batch processing capabilities with progress tracking,
    error handling, and parallel processing support.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        save_interval: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of worker threads for parallel processing
            progress_callback: Function to call with progress updates
            error_callback: Function to call when errors occur
            save_interval: Save results every N batches (0 to disable)
            output_dir: Directory to save intermediate results
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.error_callback = error_callback
        self.save_interval = save_interval
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Processing statistics
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "processing_time": 0.0,
            "batches_processed": 0,
            "errors": [],
        }
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
    def process_iterator(
        self,
        data_iterator: Iterator[Any],
        process_func: Callable,
        parallel: bool = False,
        **process_kwargs
    ) -> Iterator[Any]:
        """Process data from an iterator in batches.
        
        Args:
            data_iterator: Iterator over input data
            process_func: Function to apply to each batch or item
            parallel: Whether to use parallel processing
            **process_kwargs: Additional arguments for process_func
            
        Yields:
            Processed results
        """
        start_time = time.time()
        batch = []
        all_results = []
        
        try:
            for item in data_iterator:
                batch.append(item)
                self.stats["total_items"] += 1
                
                if len(batch) >= self.batch_size:
                    # Process current batch
                    batch_results = self._process_batch(
                        batch, process_func, parallel, **process_kwargs
                    )
                    
                    # Yield results
                    for result in batch_results:
                        yield result
                        
                    all_results.extend(batch_results)
                    
                    # Update statistics
                    self.stats["batches_processed"] += 1
                    self.stats["processed_items"] += len(batch_results)
                    
                    # Save intermediate results if configured
                    if self.save_interval and self.stats["batches_processed"] % self.save_interval == 0:
                        self._save_intermediate_results(all_results)
                        
                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(self.stats.copy())
                        
                    # Reset batch
                    batch = []
                    
            # Process final batch
            if batch:
                batch_results = self._process_batch(
                    batch, process_func, parallel, **process_kwargs
                )
                
                for result in batch_results:
                    yield result
                    
                all_results.extend(batch_results)
                self.stats["batches_processed"] += 1
                self.stats["processed_items"] += len(batch_results)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            if self.error_callback:
                self.error_callback(e)
            raise
            
        finally:
            self.stats["processing_time"] = time.time() - start_time
            
    def process_list(
        self,
        data_list: List[Any],
        process_func: Callable,
        parallel: bool = False,
        **process_kwargs
    ) -> List[Any]:
        """Process a list of data in batches.
        
        Args:
            data_list: List of input data
            process_func: Function to apply to each batch or item
            parallel: Whether to use parallel processing
            **process_kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        results = list(self.process_iterator(
            iter(data_list), process_func, parallel, **process_kwargs
        ))
        return results
        
    def _process_batch(
        self,
        batch: List[Any],
        process_func: Callable,
        parallel: bool = False,
        **process_kwargs
    ) -> List[Any]:
        """Process a single batch of data.
        
        Args:
            batch: Batch of data to process
            process_func: Function to apply to batch or items
            parallel: Whether to use parallel processing
            **process_kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        try:
            if parallel and self.max_workers and self.max_workers > 1:
                return self._process_batch_parallel(batch, process_func, **process_kwargs)
            else:
                return self._process_batch_sequential(batch, process_func, **process_kwargs)
                
        except Exception as e:
            error_info = {
                "error": str(e),
                "batch_size": len(batch),
                "timestamp": time.time(),
            }
            self.stats["errors"].append(error_info)
            self.stats["failed_items"] += len(batch)
            
            if self.error_callback:
                self.error_callback(e)
                
            # Return empty results or original batch depending on use case
            return []
            
    def _process_batch_sequential(
        self,
        batch: List[Any],
        process_func: Callable,
        **process_kwargs
    ) -> List[Any]:
        """Process batch sequentially.
        
        Args:
            batch: Batch of data to process
            process_func: Function to apply
            **process_kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        results = []
        
        for item in batch:
            try:
                if callable(process_func):
                    result = process_func(item, **process_kwargs)
                    results.append(result)
                else:
                    results.append(item)
            except Exception as e:
                logger.warning(f"Failed to process item: {e}")
                self.stats["failed_items"] += 1
                
                # Optionally include failed item with error info
                error_result = {
                    "original_item": item,
                    "error": str(e),
                    "processing_failed": True,
                }
                results.append(error_result)
                
        return results
        
    def _process_batch_parallel(
        self,
        batch: List[Any],
        process_func: Callable,
        **process_kwargs
    ) -> List[Any]:
        """Process batch in parallel using ThreadPoolExecutor.
        
        Args:
            batch: Batch of data to process
            process_func: Function to apply
            **process_kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        results = [None] * len(batch)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, item in enumerate(batch):
                future = executor.submit(process_func, item, **process_kwargs)
                future_to_index[future] = i
                
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.warning(f"Failed to process item {index}: {e}")
                    self.stats["failed_items"] += 1
                    
                    # Include error info
                    error_result = {
                        "original_item": batch[index],
                        "error": str(e),
                        "processing_failed": True,
                    }
                    results[index] = error_result
                    
        return results
        
    def _save_intermediate_results(self, results: List[Any]):
        """Save intermediate results to file.
        
        Args:
            results: Results to save
        """
        if not self.output_dir:
            return
            
        timestamp = int(time.time())
        filename = f"intermediate_results_{timestamp}.jsonl"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
                    
            logger.info(f"Saved {len(results)} intermediate results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
            
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary with progress statistics
        """
        progress = self.stats.copy()
        
        if progress["total_items"] > 0:
            progress["completion_rate"] = progress["processed_items"] / progress["total_items"]
            progress["error_rate"] = progress["failed_items"] / progress["total_items"]
        else:
            progress["completion_rate"] = 0.0
            progress["error_rate"] = 0.0
            
        if progress["processing_time"] > 0:
            progress["items_per_second"] = progress["processed_items"] / progress["processing_time"]
        else:
            progress["items_per_second"] = 0.0
            
        return progress
        
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "processing_time": 0.0,
            "batches_processed": 0,
            "errors": [],
        }


def create_data_pipeline(
    loader,
    preprocessor=None,
    validator=None,
    batch_processor=None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Create a complete data processing pipeline.
    
    Args:
        loader: Data loader instance
        preprocessor: Text preprocessor instance (optional)
        validator: Data validator instance (optional)
        batch_processor: Batch processor instance (optional)
        output_path: Path to save processed data (optional)
        
    Returns:
        Dictionary with pipeline results and statistics
    """
    if batch_processor is None:
        batch_processor = BatchProcessor()
        
    pipeline_stats = {
        "total_loaded": 0,
        "total_preprocessed": 0,
        "total_validated": 0,
        "validation_failures": 0,
        "total_saved": 0,
        "processing_time": 0.0,
    }
    
    start_time = time.time()
    processed_data = []
    
    try:
        # Load data
        data_iterator = loader.load()
        
        def pipeline_process_func(item):
            """Combined processing function for the pipeline."""
            result = {"original": item}
            
            # Preprocessing
            if preprocessor:
                try:
                    processed_item = preprocessor.preprocess_sample(item)
                    result["processed"] = processed_item
                    pipeline_stats["total_preprocessed"] += 1
                except Exception as e:
                    logger.warning(f"Preprocessing failed for item {item.get('id', 'unknown')}: {e}")
                    result["preprocessing_error"] = str(e)
                    result["processed"] = item
                    
            # Validation
            if validator:
                try:
                    validation_result = validator.validate_sample(result.get("processed", item))
                    result["validation"] = validation_result
                    pipeline_stats["total_validated"] += 1
                    
                    if not validation_result["is_valid"]:
                        pipeline_stats["validation_failures"] += 1
                except Exception as e:
                    logger.warning(f"Validation failed for item {item.get('id', 'unknown')}: {e}")
                    result["validation_error"] = str(e)
                    
            return result
            
        # Process through pipeline
        pipeline_results = list(batch_processor.process_iterator(
            data_iterator, pipeline_process_func
        ))
        
        processed_data.extend(pipeline_results)
        pipeline_stats["total_loaded"] = len(pipeline_results)
        
        # Save results if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in pipeline_results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
                    
            pipeline_stats["total_saved"] = len(pipeline_results)
            logger.info(f"Saved {len(pipeline_results)} processed samples to {output_path}")
            
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise
        
    finally:
        pipeline_stats["processing_time"] = time.time() - start_time
        
    # Combine statistics
    batch_stats = batch_processor.get_progress_info()
    pipeline_stats.update(batch_stats)
    
    return {
        "data": processed_data,
        "statistics": pipeline_stats,
        "loader_info": loader.get_dataset_info() if hasattr(loader, 'get_dataset_info') else {},
        "validation_report": validator.get_validation_report() if validator else {},
    }