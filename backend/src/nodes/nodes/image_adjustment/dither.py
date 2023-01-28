from __future__ import annotations

from enum import Enum

import numpy as np

from . import category as ImageAdjustmentCategory
from ...impl.dithering.color_distance import (
    ColorDistanceFunction, batch_nearest_palette_color,
)
from ...impl.dithering.diffusion import (
    uniform_error_diffusion_dither,
    ErrorDiffusionMap,
    ERROR_PROPAGATION_MAP_LABELS,
    nearest_color_error_diffusion_dither,
)
from ...impl.dithering.ordered import ThresholdMap, THRESHOLD_MAP_LABELS, ordered_dither
from ...impl.dithering.palette import distinct_colors, kmeans_palette
from ...impl.dithering.quantize import batch_nearest_uniform_color
from ...impl.dithering.riemersma import riemersma_dither, nearest_color_riemersma_dither
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import ImageInput, NumberInput, EnumInput, SliderInput
from ...properties.outputs import ImageOutput


class UniformDitherAlgorithm(Enum):
    NONE = "None"
    ORDERED = "Ordered"
    DIFFUSION = "Diffusion"
    RIEMERSMA = "Riemersma"


UNIFORM_DITHER_ALGORITHM_LABELS = {
    UniformDitherAlgorithm.NONE: "No dithering",
    UniformDitherAlgorithm.ORDERED: "Ordered Dithering",
    UniformDitherAlgorithm.DIFFUSION: "Error Diffusion",
    UniformDitherAlgorithm.RIEMERSMA: "Riemersma Dithering",
}


@NodeFactory.register("chainner:image:dither")
class DitherNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Apply one of a variety of dithering algorithms with a uniform (evenly-spaced) palette."
        self.inputs = [
            ImageInput(),
            NumberInput("Colors per channel", minimum=2, default=8),
            EnumInput(
                UniformDitherAlgorithm,
                option_labels=UNIFORM_DITHER_ALGORITHM_LABELS,
                default_value=UniformDitherAlgorithm.DIFFUSION,
            ).with_id(2),
            group(
                "conditional-enum",
                {
                    "enum": 2,
                    "conditions": [
                        UniformDitherAlgorithm.ORDERED.value,
                        UniformDitherAlgorithm.DIFFUSION.value,
                        UniformDitherAlgorithm.RIEMERSMA.value,
                    ],
                },
            )(
                EnumInput(
                    ThresholdMap,
                    option_labels=THRESHOLD_MAP_LABELS,
                    default_value=ThresholdMap.BAYER_16,
                ).with_id(3),
                EnumInput(
                    ErrorDiffusionMap,
                    option_labels=ERROR_PROPAGATION_MAP_LABELS,
                    default_value=ErrorDiffusionMap.FLOYD_STEINBERG,
                ).with_id(4),
                NumberInput(
                    "History Length",
                    minimum=2,
                    default=16,
                ).with_id(5),
            ),
        ]
        self.outputs = [ImageOutput(image_type="Input0")]
        self.category = ImageAdjustmentCategory
        self.name = "Dither"
        self.icon = "MdShowChart"
        self.sub = "Adjustments"

    def run(
        self,
        img: np.ndarray,
        num_colors: int,
        dither_algorithm: UniformDitherAlgorithm,
        threshold_map: ThresholdMap,
        error_diffusion_map: ErrorDiffusionMap,
        history_length: int,
    ) -> np.ndarray:
        if dither_algorithm == UniformDitherAlgorithm.NONE:
            return batch_nearest_uniform_color(img, num_colors=num_colors)
        elif dither_algorithm == UniformDitherAlgorithm.ORDERED:
            return ordered_dither(
                img, num_colors=num_colors, threshold_map=threshold_map
            )
        elif dither_algorithm == UniformDitherAlgorithm.DIFFUSION:
            return uniform_error_diffusion_dither(
                img, num_colors=num_colors, error_diffusion_map=error_diffusion_map
            )
        elif dither_algorithm == UniformDitherAlgorithm.RIEMERSMA:
            return riemersma_dither(
                img,
                num_colors=num_colors,
                history_length=history_length,
                decay_ratio=1 / history_length,
            )


class PaletteDitherAlgorithm(Enum):
    NONE = "None"
    DIFFUSION = "Diffusion"
    RIEMERSMA = "Riemersma"


PALETTE_DITHER_ALGORITHM_LABELS = {
    PaletteDitherAlgorithm.NONE: "No dithering",
    PaletteDitherAlgorithm.DIFFUSION: "Error Diffusion",
    PaletteDitherAlgorithm.RIEMERSMA: "Riemersma Dithering",
}


@NodeFactory.register("chainner:image:palette_dither")
class PaletteDitherNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Apply one of a variety of dithering algorithms using colors from a given palette. (Only the top row of pixels (y=0) of the palette will be used.)"
        self.inputs = [
            ImageInput(),
            ImageInput(label="LUT", image_type=expression.Image(channels_as="Input0")),
            EnumInput(
                PaletteDitherAlgorithm,
                option_labels=PALETTE_DITHER_ALGORITHM_LABELS,
                default_value=PaletteDitherAlgorithm.DIFFUSION,
            ).with_id(2),
            group(
                "conditional-enum",
                {
                    "enum": 2,
                    "conditions": [
                        PaletteDitherAlgorithm.DIFFUSION.value,
                        PaletteDitherAlgorithm.RIEMERSMA.value,
                    ],
                },
            )(
                EnumInput(
                    ErrorDiffusionMap,
                    option_labels=ERROR_PROPAGATION_MAP_LABELS,
                    default_value=ErrorDiffusionMap.FLOYD_STEINBERG,
                ).with_id(3),
                NumberInput(
                    "History Length",
                    minimum=2,
                    default=16,
                ).with_id(4),
            ),
        ]
        self.outputs = [ImageOutput(image_type="Input0")]
        self.category = ImageAdjustmentCategory
        self.name = "Dither (Palette)"
        self.icon = "MdShowChart"
        self.sub = "Adjustments"

    def run(
        self,
        img: np.ndarray,
        palette: np.ndarray,
        dither_algorithm: PaletteDitherAlgorithm,
        error_diffusion_map: ErrorDiffusionMap,
        history_length: int,
    ) -> np.ndarray:
        if dither_algorithm == PaletteDitherAlgorithm.NONE:
            return batch_nearest_palette_color(
                img, palette=palette, color_distance_function=ColorDistanceFunction.EUCLIDEAN
            )
        elif dither_algorithm == PaletteDitherAlgorithm.DIFFUSION:
            return nearest_color_error_diffusion_dither(
                img,
                palette=palette,
                color_distance_function=ColorDistanceFunction.EUCLIDEAN,
                error_diffusion_map=error_diffusion_map,
            )
        elif dither_algorithm == PaletteDitherAlgorithm.RIEMERSMA:
            return nearest_color_riemersma_dither(
                img,
                palette,
                color_distance_function=ColorDistanceFunction.EUCLIDEAN,
                history_length=history_length,
                decay_ratio=1 / history_length,
            )


class PaletteExtractionMethod(Enum):
    ALL="all"
    KMEANS="k-means"


PALETTE_EXTRACTION_METHOD_LABELS = {
    PaletteExtractionMethod.ALL: "All distinct colors",
    PaletteExtractionMethod.KMEANS: "K-Means",
}


@NodeFactory.register("chainner:image:palette_from_image")
class PaletteFromImage(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Create a palette an image."
        self.inputs = [
            ImageInput(),
            EnumInput(
                PaletteExtractionMethod,
                option_labels=PALETTE_EXTRACTION_METHOD_LABELS,
                default_value=PaletteExtractionMethod.KMEANS,
            ).with_id(1),
            group(
                "conditional-enum",
                {
                    "enum": 1,
                    "conditions": [
                        PaletteExtractionMethod.KMEANS.value,
                    ],
                },
            )(
                NumberInput(
                    "Palette Size",
                    minimum=2,
                    default=8,
                ).with_id(2),
            )
        ]
        self.outputs = [ImageOutput(image_type=expression.Image(channels_as="Input0"))]
        self.category = ImageAdjustmentCategory
        self.name = "Palette from Image"
        self.icon = "MdShowChart"
        self.sub = "Adjustments"

    def run(self, img: np.ndarray, palette_extraction_method: PaletteExtractionMethod, palette_size: int) -> np.ndarray:
        if palette_extraction_method == PaletteExtractionMethod.ALL:
            return distinct_colors(img)
        elif palette_extraction_method == PaletteExtractionMethod.KMEANS:
            return kmeans_palette(img, palette_size)
