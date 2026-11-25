"""Shape and execution settings for the dual-stream shifted attention encoder."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    patch_width: int = 8
    stream_width: int = 192
    blocks: int = 8
    heads: int = 6
    shifts: tuple[int, ...] = (0, -1, 1, -2, 2, -4)
    root_seeds: int = 4
    point_seeds: int = 2
    variant: str = "full"
    common_support: bool = False
    linear_tokenizer: bool = False
    repeat_patch_points: bool = False
    aux_reconstruction: bool = False
    patch_chunk_size: int = 8
    source_chunk_size: int = 128
    checkpoint_blocks: bool = True
    checkpoint_decoder: bool = True

    def __post_init__(self):
        object.__setattr__(self, "shifts", tuple(self.shifts))
        if min(self.patch_width, self.stream_width, self.blocks, self.heads,
               self.root_seeds, self.point_seeds, self.patch_chunk_size, self.source_chunk_size) < 1:
            raise ValueError("Widths, depths, head/seed counts and chunk sizes must be positive.")
        if self.stream_width % self.heads or self.stream_width % 2:
            raise ValueError("stream_width must be even and divisible by heads.")
        if len(self.shifts) != self.heads or self.shifts[0] != 0 or len(set(self.shifts)) != self.heads:
            raise ValueError("One distinct shift per head is required, with the zero shift first.")
        if any(type(delta) is not int for delta in self.shifts):
            raise ValueError("Shifts count integer patch positions.")
        if self.variant not in {"full", "synchronous", "hard_shifts", "no_gate", "feedback", "flat"}:
            raise ValueError(f"Unknown variant: {self.variant}")

    @property
    def final_width(self):
        return 2 * self.stream_width
