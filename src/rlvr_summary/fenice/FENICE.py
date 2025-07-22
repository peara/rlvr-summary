from typing import Dict, List, Optional
import logging

import numpy as np
import torch
from tqdm import tqdm

from .model_manager import model_manager
from .utils.utils import split_into_paragraphs, split_into_sentences_batched


class FENICE:
    def __init__(
        self,
        use_coref: bool = False,
        num_sent_per_paragraph: int = 5,
        sliding_paragraphs=True,
        sliding_stride: int = 1,
        doc_level_nli=True,
        paragraph_level_nli=True,
        claim_extractor_batch_size: int = 256,
        coreference_batch_size: int = 1,
        nli_batch_size: int = 256,
        nli_max_length: int = 1024,
    ) -> None:
        self.num_sent_per_paragraph = num_sent_per_paragraph
        self.claim_extractor_batch_size = claim_extractor_batch_size
        self.coreference_batch_size = coreference_batch_size
        self.nli_batch_size = nli_batch_size
        self.sliding_paragraphs = sliding_paragraphs
        self.sliding_stride = sliding_stride
        self.sentences_cache = {}
        self.coref_clusters_cache = {}
        self.claims_cache = {}
        self.alignments_cache = {}
        self.use_coref = use_coref
        self.doc_level_nli = doc_level_nli
        self.paragraph_level_nli = paragraph_level_nli
        # Remove individual model loading - use model manager instead
        self.nli_max_length = nli_max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def clear_model_cache(self):
        """Clear cached models to free memory if needed."""
        model_manager.clear_cache()

    def get_model_info(self):
        """Get information about loaded models."""
        return model_manager.get_memory_info()

    def _score(self, sample_id: int, document: str, summary: str):
        doc_id = self.get_id(sample_id, document)
        sentences_offsets = self.sentences_cache[doc_id]
        sentences = [s[0] for s in sentences_offsets]
        offsets = [(s[1], s[2]) for s in sentences_offsets]
        # paragraphs are sliding windows of {num_sent_per_paragraph} sentences
        paragraphs = split_into_paragraphs(
            sentences,
            self.num_sent_per_paragraph,
            sliding_paragraphs=self.sliding_paragraphs,
            sliding_stride=self.sliding_stride,
        )
        # claim extraction
        summary_id = self.get_id(sample_id, summary)
        summary_claims = self.claims_cache.get(summary_id, [summary])
        alignments = []
        for claim_id, claim in enumerate(summary_claims):
            # sentence-level alignment
            sentence_level_alignment = self.get_alignment(
                premises=sentences,
                hypothesis=claim,
                sample_id=sample_id,
                hypothesis_id=claim_id,
            )
            coref_alignment = None
            if self.use_coref:
                coref_clusters = self.coref_clusters_cache[doc_id]
                # Get coreference model from manager
                coref_model = model_manager.get_coref_model(
                    batch_size=self.coreference_batch_size, 
                    device=self.device
                )
                # modified versions of {aligned_sentence} obtained through coreference resolution
                coref_premises = coref_model.get_coref_versions(
                    sentence=sentence_level_alignment["source_passage"],
                    text=document,
                    sentences=sentences,
                    offsets=offsets,
                    clusters=coref_clusters,
                )
                if coref_premises:
                    coref_alignment = self.get_alignment(
                        sample_id=sample_id,
                        hypothesis_id=claim_id,
                        hypothesis=claim,
                        premises=coref_premises,
                        alignment_prefix="coref",
                    )
            paragraph_level_alignment = None
            if len(paragraphs) > 1 and self.paragraph_level_nli:
                paragraph_level_alignment = self.get_alignment(
                    premises=paragraphs,
                    hypothesis=claim,
                    sample_id=sample_id,
                    hypothesis_id=claim_id,
                    alignment_prefix="par",
                )
            doc_level_alignment = None
            if self.doc_level_nli and len(paragraphs) > 1:
                doc_level_alignment = self.get_alignment(
                    hypothesis=claim,
                    premises=[document],
                    sample_id=sample_id,
                    hypothesis_id=claim_id,
                    alignment_prefix="doc",
                )
                doc_level_alignment["source_passage"] = "DOCUMENT"
            sample_alignments = [
                sentence_level_alignment,
                coref_alignment,
                paragraph_level_alignment,
                doc_level_alignment,
            ]
            alignment = self.max_alignment(sample_alignments)
            alignments.append(alignment)
        score = np.mean([al["score"] for al in alignments])
        return {"score": score, "alignments": alignments}

    def max_alignment(self, sample_alignments):
        sample_alignments = [s for s in sample_alignments if s is not None]
        alignment = max(sample_alignments, key=lambda x: x["score"])
        return alignment

    def get_alignment(
        self,
        premises,
        hypothesis,
        sample_id,
        hypothesis_id,
        alignment_prefix: Optional[str] = None,
    ):
        alignments_ids = []
        for prem_id in range(len(premises)):
            pair_id = self.get_alignment_id(
                sample_id=sample_id,
                premise_id=prem_id,
                hypothesis_id=hypothesis_id,
                premise=premises[prem_id],
                alignment_prefix=alignment_prefix,
            )
            alignments_ids.append(pair_id)
        loaded_alignment = self.load_alignment(
            alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
        )
        if loaded_alignment is None:
            all_pairs = [(x, hypothesis) for x in premises]
            self.cache_alignment(alignments_ids, all_pairs)
            alignment, _ = self.load_alignment(
                alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
            )
        else:
            alignment, _ = loaded_alignment
        return alignment

    def score_batch(self, batch: List[Dict[str, str]], document_cache_data=None) -> List[Dict]:
        """Score a batch of document-summary pairs, using pre-cached data when available."""
        documents = [el["document"] for el in batch]
        summaries = [el["summary"] for el in batch]
        self.cache(documents, summaries, document_cache_data)
        predictions = []
        for sample_id, (doc, summary) in tqdm(
            enumerate(zip(documents, summaries)),
            total=len(documents),
            desc="Computing FENICE...",
        ):
            predictions.append(self._score(sample_id, doc, summary))

        # Don't delete the NLI aligner - keep it cached for reuse
        torch.cuda.empty_cache()

        return predictions

    def cache(self, documents, summaries, document_cache_data=None):
        """Cache document and summary data, using pre-cached data when available."""
        self.cache_sentences(documents, document_cache_data)
        if self.use_coref:
            self.cache_coref(documents, document_cache_data)
        self.cache_claims(summaries)
        self.cache_alignments(documents, summaries)

    def cache_sentences(self, documents, document_cache_data=None):
        """Cache sentences, using pre-cached data when available."""
        if document_cache_data:
            # Use pre-cached sentence data
            logger = logging.getLogger(__name__)
            logger.info("Using pre-cached sentence data from document cache")
            
            for i, doc in enumerate(documents):
                doc_id = self.get_id(i, doc)
                
                # Look for cached data for this document
                if i in document_cache_data:
                    cached_doc = document_cache_data[i]
                    if 'sentences' in cached_doc:
                        self.sentences_cache[doc_id] = cached_doc['sentences']
                        continue
                
                # Fallback to runtime computation if cache missing
                logger.warning(f"No cached sentences for document {i}, computing at runtime")
                doc_sentences = split_into_sentences_batched(
                    [doc], batch_size=512, return_offsets=True
                )[0]
                self.sentences_cache[doc_id] = doc_sentences
        else:
            # Original behavior - compute sentences at runtime
            all_sentences = split_into_sentences_batched(
                documents, batch_size=512, return_offsets=True
            )
            for i, sentences in enumerate(all_sentences):
                id = self.get_id(i, documents[i])
                self.sentences_cache[id] = sentences

    def get_docs_to_process(self, documents, cache):
        ids = [doc_id for doc_id in range(len(documents))]
        ids_to_process = [
            id for id in ids if self.get_id(id, documents[id]) not in cache
        ]
        doc_to_process = [documents[i] for i in ids_to_process]
        return doc_to_process, ids_to_process

    def get_id(self, sample_id: int, text: str, k_chars: int = 100):
        id = f"{sample_id}{text[:k_chars]}"
        return id

    # cache claim extraction outputs
    def cache_claims(self, summaries):
        claim_extractor = model_manager.get_claim_extractor(
            batch_size=self.claim_extractor_batch_size, device=self.device
        )
        claims_predictions = claim_extractor.process_batch(summaries)
        for summ_id, claims in enumerate(claims_predictions):
            id = self.get_id(summ_id, summaries[summ_id])
            self.claims_cache[id] = claims
        # Don't delete - keep models cached for reuse
        torch.cuda.empty_cache()

    # cache coreference resolution outputs
    def cache_coref(self, documents, document_cache_data=None):
        """Cache coreference resolution, using pre-cached data when available."""
        if document_cache_data:
            # Use pre-cached coreference data
            logger = logging.getLogger(__name__)
            logger.info("Checking for pre-cached coreference data")
            
            docs_to_process = []
            docs_indices = []
            
            for i, doc in enumerate(documents):
                doc_id = self.get_id(i, doc)
                
                # Check if we have cached coreference data
                if i in document_cache_data:
                    cached_doc = document_cache_data[i]
                    if 'coref_clusters' in cached_doc:
                        self.coref_clusters_cache[doc_id] = cached_doc['coref_clusters']
                        continue
                
                # Need to compute for this document
                docs_to_process.append(doc)
                docs_indices.append(i)
            
            if docs_to_process:
                logger.info(f"Computing coreference for {len(docs_to_process)} documents at runtime")
                coref_model = model_manager.get_coref_model(
                    batch_size=self.coreference_batch_size, device=self.device
                )
                clusters_batch = coref_model.get_clusters_batch(docs_to_process)
                
                for idx, clusters in zip(docs_indices, clusters_batch):
                    doc_id = self.get_id(idx, documents[idx])
                    self.coref_clusters_cache[doc_id] = clusters
        else:
            # Original behavior - compute all at runtime
            coref_model = model_manager.get_coref_model(
                batch_size=self.coreference_batch_size, device=self.device
            )
            all_clusters = coref_model.get_clusters_batch(documents)
            for doc_id, clusters in enumerate(all_clusters):
                id = self.get_id(doc_id, documents[doc_id])
                self.coref_clusters_cache[id] = clusters
        # Don't delete - keep models cached for reuse
        torch.cuda.empty_cache()

    def cache_alignments(self, documents: List[str], summaries: List[str]):
        alignments_ids, all_pairs = [], []
        alignment_ids_doc, all_pairs_doc = [], []
        self.nli_aligner = model_manager.get_nli_aligner(
            batch_size=self.nli_batch_size,
            device=self.device,
            max_length=self.nli_max_length,
        )
        for sample_id in range(len(summaries)):
            summary_id = self.get_id(sample_id, summaries[sample_id])
            document_id = self.get_id(sample_id, documents[sample_id])
            claims = self.claims_cache[summary_id]
            sentences = [s[0] for s in self.sentences_cache[document_id]]
            paragraphs = split_into_paragraphs(
                sentences,
                self.num_sent_per_paragraph,
                sliding_paragraphs=self.sliding_paragraphs,
                sliding_stride=self.sliding_stride,
            )
            alignments_ids, all_pairs = self.compute_nli_pairs(
                alignments_ids, all_pairs, claims, sample_id, sentences
            )
            if self.paragraph_level_nli:
                alignments_ids, all_pairs = self.compute_nli_pairs(
                    alignments_ids,
                    all_pairs,
                    claims,
                    sample_id,
                    paragraphs,
                    prefix="par",
                )
            if self.doc_level_nli:
                alignment_ids_doc, all_pairs_doc = self.compute_nli_pairs(
                    alignment_ids_doc,
                    all_pairs_doc,
                    claims,
                    sample_id,
                    [documents[sample_id]],
                    prefix="doc",
                )
        self.cache_alignment(alignments_ids, all_pairs)
        self.nli_aligner.batch_size = 1
        self.nli_aligner.max_length = 4096
        self.cache_alignment(alignment_ids_doc, all_pairs_doc)
        self.nli_aligner.batch_size = self.nli_batch_size
        self.nli_max_length = self.nli_max_length

    def compute_nli_pairs(
        self, alignments_ids, all_pairs, claims, sample_id, premises, prefix=None
    ):
        for premise_id, premise in enumerate(premises):
            for claim_id, claim in enumerate(claims):
                alignment_id = self.get_alignment_id(
                    sample_id, premise_id, claim_id, premise, prefix
                )
                alignments_ids.append(alignment_id)
                all_pairs.append((premise, claim))
        return alignments_ids, all_pairs

    def get_alignment_id(
        self,
        sample_id: int,
        premise_id: int,
        hypothesis_id: int,
        premise: str,
        alignment_prefix: Optional[str] = None,
        k_chars: int = 100,
    ):
        id = f"{sample_id}-{premise_id}-{hypothesis_id}-{premise[:k_chars]}"
        if alignment_prefix is not None:
            id = f"{alignment_prefix}{id}"
        return id

    def cache_alignment(
        self, alignments_ids, all_pairs, disable_prog_bar: bool = False
    ):
        probabilities = self.nli_aligner.process_batch(
            all_pairs, disable_prog_bar=disable_prog_bar
        )
        for i, id in enumerate(alignments_ids):
            ent, contr, neut = (
                probabilities[0][i],
                probabilities[1][i],
                probabilities[2][i],
            )
            self.alignments_cache[id] = (ent, contr, neut)

    def load_alignment(
        self, alignments_ids: List[str], premises: List[str], hypothesis: str
    ):
        if all([k in self.alignments_cache for k in alignments_ids]):
            scores = [self.alignments_cache[key] for key in alignments_ids]
            alignment = None
            max_score = -np.inf
            for i, (ent, contr, neut) in enumerate(scores):
                ent, contr, neut = ent.item(), contr.item(), neut.item()
                align_score = ent - neut
                if align_score > max_score:
                    max_score = align_score
                    alignment = (premises[i], [ent, contr, neut])
            return (
                {
                    "score": max_score,
                    "summary_claim": hypothesis,
                    "source_passage": alignment[0],
                },
                scores,
            )
        else:
            return None
