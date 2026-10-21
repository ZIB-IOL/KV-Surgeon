"""
Chunking strategies for long contexts.

- LongHealthChunking: split on <text_0> tags, one chunk per note chain.
- LQAChunking: split on [start of {filename}] markers, one chunk per code file.
- LQAPackedChunking: LQAChunking with consecutive files packed up to max_chunk_size tokens.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Chunk:
    """A chunk of context text with its approximate token span in the full context."""
    text: str
    start_token_idx: int
    end_token_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """Split the text into chunks; the tokenizer is used for token positions."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LongHealthChunking(ChunkingStrategy):
    """Split article on <text_0> tags, grouping all notes until the next <text_0>."""

    @property
    def name(self) -> str:
        return "longhealth"

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        split_pattern = r'(?=<text_0>)'
        parts = re.split(split_pattern, article_text)

        chunks = []
        cumulative_len = 0

        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                cumulative_len += len(part)
                continue

            if part_stripped.startswith('<text_0>'):
                chunk_tokens = tokenizer.encode(part_stripped, add_special_tokens=False)
                prefix_tokens = tokenizer.encode(article_text[:cumulative_len], add_special_tokens=False)

                note_ids = re.findall(r'<text_(\d+)>', part_stripped)
                max_note = max(int(n) for n in note_ids) if note_ids else 0

                chunks.append(Chunk(
                    text=part_stripped,
                    start_token_idx=len(prefix_tokens),
                    end_token_idx=len(prefix_tokens) + len(chunk_tokens),
                    metadata={
                        'note_range': f'text_0-text_{max_note}',
                        'num_notes': len(note_ids)
                    }
                ))

            cumulative_len += len(part)

        # No <text_0> tags: the whole text is one chunk.
        if not chunks:
            tokens = tokenizer.encode(article_text, add_special_tokens=False)
            chunks.append(Chunk(
                text=article_text,
                start_token_idx=0,
                end_token_idx=len(tokens),
                metadata={'note_range': None, 'num_notes': 0}
            ))

        return chunks


class LQAChunking(ChunkingStrategy):
    """Split article on [start of {filename}] markers (one chunk per code file)."""

    def __init__(self, max_chunk_size: int = 10000):
        """Files longer than max_chunk_size tokens are split into several chunks."""
        self.max_chunk_size = max_chunk_size

    @property
    def name(self) -> str:
        return "lqa"

    def _split_large_chunk(self, text: str, filename: str, start_token_idx: int,
                           tokenizer) -> List[Chunk]:
        """Split a chunk that exceeds max_chunk_size into smaller pieces."""
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_chunk_size:
            return [Chunk(
                text=text,
                start_token_idx=start_token_idx,
                end_token_idx=start_token_idx + len(tokens),
                metadata={'filename': filename}
            )]

        chunks = []
        for i in range(0, len(tokens), self.max_chunk_size):
            chunk_tokens = tokens[i:i + self.max_chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            part_num = i // self.max_chunk_size + 1
            total_parts = (len(tokens) + self.max_chunk_size - 1) // self.max_chunk_size

            chunks.append(Chunk(
                text=chunk_text,
                start_token_idx=start_token_idx + i,
                end_token_idx=start_token_idx + i + len(chunk_tokens),
                metadata={
                    'filename': filename,
                    'part': part_num,
                    'total_parts': total_parts
                }
            ))

        return chunks

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """Split on the [start of path/to/file.py] markers that open each file."""
        pattern = r'(\[start of [^\]]+\])'
        parts = re.split(pattern, article_text)

        chunks = []
        current_chunk_text = ""
        current_filename = None
        cumulative_char_pos = 0

        for part in parts:
            if part.startswith('[start of '):
                if current_chunk_text.strip():
                    prefix_tokens = tokenizer.encode(
                        article_text[:cumulative_char_pos - len(current_chunk_text)],
                        add_special_tokens=False
                    )

                    chunks.extend(self._split_large_chunk(
                        current_chunk_text,
                        current_filename,
                        len(prefix_tokens),
                        tokenizer
                    ))

                filename_match = re.search(r'\[start of (.+)\]', part)
                current_filename = filename_match.group(1) if filename_match else None
                current_chunk_text = part
            else:
                current_chunk_text += part

            cumulative_char_pos += len(part)

        if current_chunk_text.strip():
            prefix_tokens = tokenizer.encode(
                article_text[:cumulative_char_pos - len(current_chunk_text)],
                add_special_tokens=False
            )

            chunks.extend(self._split_large_chunk(
                current_chunk_text,
                current_filename,
                len(prefix_tokens),
                tokenizer
            ))

        if not chunks:
            chunks.extend(self._split_large_chunk(
                article_text,
                None,
                0,
                tokenizer
            ))

        return chunks


class LQAPackedChunking(LQAChunking):
    """Per-file chunks, with consecutive files packed into chunks of at most max_chunk_size tokens.

    Avoids one compaction (and query-generation pass) per small file. File boundaries are
    respected; oversized files are split as in LQAChunking.
    """

    @property
    def name(self) -> str:
        return "lqa_pack"

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        packed: List[Chunk] = []
        cur: Optional[Chunk] = None
        for c in super().chunk(article_text, tokenizer):
            n = c.end_token_idx - c.start_token_idx
            fname = c.metadata.get('filename')
            if cur is not None and (cur.end_token_idx - cur.start_token_idx) + n <= self.max_chunk_size:
                names = cur.metadata['filenames'] + [fname]
                cur = Chunk(
                    text=cur.text + c.text,
                    start_token_idx=cur.start_token_idx,
                    end_token_idx=c.end_token_idx,
                    metadata={'filenames': names, 'filename': f"{names[0]} .. {names[-1]} ({len(names)} files)"},
                )
            else:
                if cur is not None:
                    packed.append(cur)
                cur = Chunk(
                    text=c.text,
                    start_token_idx=c.start_token_idx,
                    end_token_idx=c.end_token_idx,
                    metadata={'filenames': [fname], 'filename': fname},
                )
        if cur is not None:
            packed.append(cur)
        return packed


def get_chunking_strategy(name: str, **kwargs) -> Optional[ChunkingStrategy]:
    """Build a chunking strategy: 'longhealth', 'lqa_pack' (uses chunk_size), or None for 'none'."""
    if name is None or name.lower() == 'none':
        return None

    name_lower = name.lower()

    if name_lower == 'longhealth':
        return LongHealthChunking()
    elif name_lower == 'lqa_pack':
        max_chunk_size = kwargs.get('chunk_size') or kwargs.get('max_chunk_size', 10000)
        return LQAPackedChunking(max_chunk_size=max_chunk_size)
    else:
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Supported strategies: 'longhealth', 'lqa_pack', 'none'"
        )
