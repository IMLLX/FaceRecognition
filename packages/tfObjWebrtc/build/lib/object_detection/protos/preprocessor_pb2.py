# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/preprocessor.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*object_detection/protos/preprocessor.proto\x12\x17object_detection.protos\"\xd5\x18\n\x11PreprocessingStep\x12\x42\n\x0fnormalize_image\x18\x01 \x01(\x0b\x32\'.object_detection.protos.NormalizeImageH\x00\x12O\n\x16random_horizontal_flip\x18\x02 \x01(\x0b\x32-.object_detection.protos.RandomHorizontalFlipH\x00\x12R\n\x18random_pixel_value_scale\x18\x03 \x01(\x0b\x32..object_detection.protos.RandomPixelValueScaleH\x00\x12G\n\x12random_image_scale\x18\x04 \x01(\x0b\x32).object_detection.protos.RandomImageScaleH\x00\x12\x46\n\x12random_rgb_to_gray\x18\x05 \x01(\x0b\x32(.object_detection.protos.RandomRGBtoGrayH\x00\x12S\n\x18random_adjust_brightness\x18\x06 \x01(\x0b\x32/.object_detection.protos.RandomAdjustBrightnessH\x00\x12O\n\x16random_adjust_contrast\x18\x07 \x01(\x0b\x32-.object_detection.protos.RandomAdjustContrastH\x00\x12\x45\n\x11random_adjust_hue\x18\x08 \x01(\x0b\x32(.object_detection.protos.RandomAdjustHueH\x00\x12S\n\x18random_adjust_saturation\x18\t \x01(\x0b\x32/.object_detection.protos.RandomAdjustSaturationH\x00\x12K\n\x14random_distort_color\x18\n \x01(\x0b\x32+.object_detection.protos.RandomDistortColorH\x00\x12I\n\x13random_jitter_boxes\x18\x0b \x01(\x0b\x32*.object_detection.protos.RandomJitterBoxesH\x00\x12\x45\n\x11random_crop_image\x18\x0c \x01(\x0b\x32(.object_detection.protos.RandomCropImageH\x00\x12\x43\n\x10random_pad_image\x18\r \x01(\x0b\x32\'.object_detection.protos.RandomPadImageH\x00\x12L\n\x15random_crop_pad_image\x18\x0e \x01(\x0b\x32+.object_detection.protos.RandomCropPadImageH\x00\x12W\n\x1brandom_crop_to_aspect_ratio\x18\x0f \x01(\x0b\x32\x30.object_detection.protos.RandomCropToAspectRatioH\x00\x12K\n\x14random_black_patches\x18\x10 \x01(\x0b\x32+.object_detection.protos.RandomBlackPatchesH\x00\x12K\n\x14random_resize_method\x18\x11 \x01(\x0b\x32+.object_detection.protos.RandomResizeMethodH\x00\x12\x61\n scale_boxes_to_pixel_coordinates\x18\x12 \x01(\x0b\x32\x35.object_detection.protos.ScaleBoxesToPixelCoordinatesH\x00\x12<\n\x0cresize_image\x18\x13 \x01(\x0b\x32$.object_detection.protos.ResizeImageH\x00\x12M\n\x15subtract_channel_mean\x18\x14 \x01(\x0b\x32,.object_detection.protos.SubtractChannelMeanH\x00\x12\x41\n\x0fssd_random_crop\x18\x15 \x01(\x0b\x32&.object_detection.protos.SSDRandomCropH\x00\x12H\n\x13ssd_random_crop_pad\x18\x16 \x01(\x0b\x32).object_detection.protos.SSDRandomCropPadH\x00\x12\x64\n\"ssd_random_crop_fixed_aspect_ratio\x18\x17 \x01(\x0b\x32\x36.object_detection.protos.SSDRandomCropFixedAspectRatioH\x00\x12k\n&ssd_random_crop_pad_fixed_aspect_ratio\x18\x18 \x01(\x0b\x32\x39.object_detection.protos.SSDRandomCropPadFixedAspectRatioH\x00\x12K\n\x14random_vertical_flip\x18\x19 \x01(\x0b\x32+.object_detection.protos.RandomVerticalFlipH\x00\x12\x46\n\x11random_rotation90\x18\x1a \x01(\x0b\x32).object_detection.protos.RandomRotation90H\x00\x12\x39\n\x0brgb_to_gray\x18\x1b \x01(\x0b\x32\".object_detection.protos.RGBtoGrayH\x00\x12_\n\x1f\x63onvert_class_logits_to_softmax\x18\x1c \x01(\x0b\x32\x34.object_detection.protos.ConvertClassLogitsToSoftmaxH\x00\x12T\n\x19random_absolute_pad_image\x18\x1d \x01(\x0b\x32/.object_detection.protos.RandomAbsolutePadImageH\x00\x12R\n\x18random_self_concat_image\x18\x1e \x01(\x0b\x32..object_detection.protos.RandomSelfConcatImageH\x00\x12\x46\n\x11\x61utoaugment_image\x18\x1f \x01(\x0b\x32).object_detection.protos.AutoAugmentImageH\x00\x12[\n\x1c\x64rop_label_probabilistically\x18  \x01(\x0b\x32\x33.object_detection.protos.DropLabelProbabilisticallyH\x00\x12<\n\x0cremap_labels\x18! \x01(\x0b\x32$.object_detection.protos.RemapLabelsH\x00\x12I\n\x13random_jpeg_quality\x18\" \x01(\x0b\x32*.object_detection.protos.RandomJpegQualityH\x00\x12\x63\n!random_downscale_to_target_pixels\x18# \x01(\x0b\x32\x36.object_detection.protos.RandomDownscaleToTargetPixelsH\x00\x12M\n\x15random_patch_gaussian\x18$ \x01(\x0b\x32,.object_detection.protos.RandomPatchGaussianH\x00\x12W\n\x1brandom_square_crop_by_scale\x18% \x01(\x0b\x32\x30.object_detection.protos.RandomSquareCropByScaleH\x00\x12\x65\n#random_scale_crop_and_pad_to_square\x18& \x01(\x0b\x32\x36.object_detection.protos.RandomScaleCropAndPadToSquareH\x00\x12<\n\x0c\x61\x64just_gamma\x18\' \x01(\x0b\x32$.object_detection.protos.AdjustGammaH\x00\x42\x14\n\x12preprocessing_step\"v\n\x0eNormalizeImage\x12\x17\n\x0foriginal_minval\x18\x01 \x01(\x02\x12\x17\n\x0foriginal_maxval\x18\x02 \x01(\x02\x12\x18\n\rtarget_minval\x18\x03 \x01(\x02:\x01\x30\x12\x18\n\rtarget_maxval\x18\x04 \x01(\x02:\x01\x31\"S\n\x14RandomHorizontalFlip\x12!\n\x19keypoint_flip_permutation\x18\x01 \x03(\x05\x12\x18\n\x0bprobability\x18\x02 \x01(\x02:\x03\x30.5\"Q\n\x12RandomVerticalFlip\x12!\n\x19keypoint_flip_permutation\x18\x01 \x03(\x05\x12\x18\n\x0bprobability\x18\x02 \x01(\x02:\x03\x30.5\"N\n\x10RandomRotation90\x12 \n\x18keypoint_rot_permutation\x18\x01 \x03(\x05\x12\x18\n\x0bprobability\x18\x02 \x01(\x02:\x03\x30.5\"A\n\x15RandomPixelValueScale\x12\x13\n\x06minval\x18\x01 \x01(\x02:\x03\x30.9\x12\x13\n\x06maxval\x18\x02 \x01(\x02:\x03\x31.1\"L\n\x10RandomImageScale\x12\x1c\n\x0fmin_scale_ratio\x18\x01 \x01(\x02:\x03\x30.5\x12\x1a\n\x0fmax_scale_ratio\x18\x02 \x01(\x02:\x01\x32\"+\n\x0fRandomRGBtoGray\x12\x18\n\x0bprobability\x18\x01 \x01(\x02:\x03\x30.1\"0\n\x16RandomAdjustBrightness\x12\x16\n\tmax_delta\x18\x01 \x01(\x02:\x03\x30.2\"G\n\x14RandomAdjustContrast\x12\x16\n\tmin_delta\x18\x01 \x01(\x02:\x03\x30.8\x12\x17\n\tmax_delta\x18\x02 \x01(\x02:\x04\x31.25\"*\n\x0fRandomAdjustHue\x12\x17\n\tmax_delta\x18\x01 \x01(\x02:\x04\x30.02\"I\n\x16RandomAdjustSaturation\x12\x16\n\tmin_delta\x18\x01 \x01(\x02:\x03\x30.8\x12\x17\n\tmax_delta\x18\x02 \x01(\x02:\x04\x31.25\",\n\x12RandomDistortColor\x12\x16\n\x0e\x63olor_ordering\x18\x01 \x01(\x05\"(\n\x11RandomJitterBoxes\x12\x13\n\x05ratio\x18\x01 \x01(\x02:\x04\x30.05\"\xeb\x01\n\x0fRandomCropImage\x12\x1d\n\x12min_object_covered\x18\x01 \x01(\x02:\x01\x31\x12\x1e\n\x10min_aspect_ratio\x18\x02 \x01(\x02:\x04\x30.75\x12\x1e\n\x10max_aspect_ratio\x18\x03 \x01(\x02:\x04\x31.33\x12\x15\n\x08min_area\x18\x04 \x01(\x02:\x03\x30.1\x12\x13\n\x08max_area\x18\x05 \x01(\x02:\x01\x31\x12\x1b\n\x0eoverlap_thresh\x18\x06 \x01(\x02:\x03\x30.3\x12\x18\n\nclip_boxes\x18\x08 \x01(\x08:\x04true\x12\x16\n\x0brandom_coef\x18\x07 \x01(\x02:\x01\x30\"\x89\x01\n\x0eRandomPadImage\x12\x18\n\x10min_image_height\x18\x01 \x01(\x05\x12\x17\n\x0fmin_image_width\x18\x02 \x01(\x05\x12\x18\n\x10max_image_height\x18\x03 \x01(\x05\x12\x17\n\x0fmax_image_width\x18\x04 \x01(\x05\x12\x11\n\tpad_color\x18\x05 \x03(\x02\"b\n\x16RandomAbsolutePadImage\x12\x1a\n\x12max_height_padding\x18\x01 \x01(\x05\x12\x19\n\x11max_width_padding\x18\x02 \x01(\x05\x12\x11\n\tpad_color\x18\x03 \x03(\x02\"\xbf\x02\n\x12RandomCropPadImage\x12\x1d\n\x12min_object_covered\x18\x01 \x01(\x02:\x01\x31\x12\x1e\n\x10min_aspect_ratio\x18\x02 \x01(\x02:\x04\x30.75\x12\x1e\n\x10max_aspect_ratio\x18\x03 \x01(\x02:\x04\x31.33\x12\x15\n\x08min_area\x18\x04 \x01(\x02:\x03\x30.1\x12\x13\n\x08max_area\x18\x05 \x01(\x02:\x01\x31\x12\x1b\n\x0eoverlap_thresh\x18\x06 \x01(\x02:\x03\x30.3\x12\x18\n\nclip_boxes\x18\x0b \x01(\x08:\x04true\x12\x16\n\x0brandom_coef\x18\x07 \x01(\x02:\x01\x30\x12\x1d\n\x15min_padded_size_ratio\x18\x08 \x03(\x02\x12\x1d\n\x15max_padded_size_ratio\x18\t \x03(\x02\x12\x11\n\tpad_color\x18\n \x03(\x02\"i\n\x17RandomCropToAspectRatio\x12\x17\n\x0c\x61spect_ratio\x18\x01 \x01(\x02:\x01\x31\x12\x1b\n\x0eoverlap_thresh\x18\x02 \x01(\x02:\x03\x30.3\x12\x18\n\nclip_boxes\x18\x03 \x01(\x08:\x04true\"o\n\x12RandomBlackPatches\x12\x1d\n\x11max_black_patches\x18\x01 \x01(\x05:\x02\x31\x30\x12\x18\n\x0bprobability\x18\x02 \x01(\x02:\x03\x30.5\x12 \n\x13size_to_image_ratio\x18\x03 \x01(\x02:\x03\x30.1\"A\n\x12RandomResizeMethod\x12\x15\n\rtarget_height\x18\x01 \x01(\x05\x12\x14\n\x0ctarget_width\x18\x02 \x01(\x05\"\x0b\n\tRGBtoGray\"\x1e\n\x1cScaleBoxesToPixelCoordinates\"\xc0\x01\n\x0bResizeImage\x12\x12\n\nnew_height\x18\x01 \x01(\x05\x12\x11\n\tnew_width\x18\x02 \x01(\x05\x12\x45\n\x06method\x18\x03 \x01(\x0e\x32+.object_detection.protos.ResizeImage.Method:\x08\x42ILINEAR\"C\n\x06Method\x12\x08\n\x04\x41REA\x10\x01\x12\x0b\n\x07\x42ICUBIC\x10\x02\x12\x0c\n\x08\x42ILINEAR\x10\x03\x12\x14\n\x10NEAREST_NEIGHBOR\x10\x04\"$\n\x13SubtractChannelMean\x12\r\n\x05means\x18\x01 \x03(\x02\"\xd3\x01\n\x16SSDRandomCropOperation\x12\x1a\n\x12min_object_covered\x18\x01 \x01(\x02\x12\x18\n\x10min_aspect_ratio\x18\x02 \x01(\x02\x12\x18\n\x10max_aspect_ratio\x18\x03 \x01(\x02\x12\x10\n\x08min_area\x18\x04 \x01(\x02\x12\x10\n\x08max_area\x18\x05 \x01(\x02\x12\x16\n\x0eoverlap_thresh\x18\x06 \x01(\x02\x12\x18\n\nclip_boxes\x18\x08 \x01(\x08:\x04true\x12\x13\n\x0brandom_coef\x18\x07 \x01(\x02\"T\n\rSSDRandomCrop\x12\x43\n\noperations\x18\x01 \x03(\x0b\x32/.object_detection.protos.SSDRandomCropOperation\"\xd3\x02\n\x19SSDRandomCropPadOperation\x12\x1a\n\x12min_object_covered\x18\x01 \x01(\x02\x12\x18\n\x10min_aspect_ratio\x18\x02 \x01(\x02\x12\x18\n\x10max_aspect_ratio\x18\x03 \x01(\x02\x12\x10\n\x08min_area\x18\x04 \x01(\x02\x12\x10\n\x08max_area\x18\x05 \x01(\x02\x12\x16\n\x0eoverlap_thresh\x18\x06 \x01(\x02\x12\x18\n\nclip_boxes\x18\r \x01(\x08:\x04true\x12\x13\n\x0brandom_coef\x18\x07 \x01(\x02\x12\x1d\n\x15min_padded_size_ratio\x18\x08 \x03(\x02\x12\x1d\n\x15max_padded_size_ratio\x18\t \x03(\x02\x12\x13\n\x0bpad_color_r\x18\n \x01(\x02\x12\x13\n\x0bpad_color_g\x18\x0b \x01(\x02\x12\x13\n\x0bpad_color_b\x18\x0c \x01(\x02\"Z\n\x10SSDRandomCropPad\x12\x46\n\noperations\x18\x01 \x03(\x0b\x32\x32.object_detection.protos.SSDRandomCropPadOperation\"\xaf\x01\n&SSDRandomCropFixedAspectRatioOperation\x12\x1a\n\x12min_object_covered\x18\x01 \x01(\x02\x12\x10\n\x08min_area\x18\x04 \x01(\x02\x12\x10\n\x08max_area\x18\x05 \x01(\x02\x12\x16\n\x0eoverlap_thresh\x18\x06 \x01(\x02\x12\x18\n\nclip_boxes\x18\x08 \x01(\x08:\x04true\x12\x13\n\x0brandom_coef\x18\x07 \x01(\x02\"\x8d\x01\n\x1dSSDRandomCropFixedAspectRatio\x12S\n\noperations\x18\x01 \x03(\x0b\x32?.object_detection.protos.SSDRandomCropFixedAspectRatioOperation\x12\x17\n\x0c\x61spect_ratio\x18\x02 \x01(\x02:\x01\x31\"\xe6\x01\n)SSDRandomCropPadFixedAspectRatioOperation\x12\x1a\n\x12min_object_covered\x18\x01 \x01(\x02\x12\x18\n\x10min_aspect_ratio\x18\x02 \x01(\x02\x12\x18\n\x10max_aspect_ratio\x18\x03 \x01(\x02\x12\x10\n\x08min_area\x18\x04 \x01(\x02\x12\x10\n\x08max_area\x18\x05 \x01(\x02\x12\x16\n\x0eoverlap_thresh\x18\x06 \x01(\x02\x12\x18\n\nclip_boxes\x18\x08 \x01(\x08:\x04true\x12\x13\n\x0brandom_coef\x18\x07 \x01(\x02\"\xd1\x01\n SSDRandomCropPadFixedAspectRatio\x12V\n\noperations\x18\x01 \x03(\x0b\x32\x42.object_detection.protos.SSDRandomCropPadFixedAspectRatioOperation\x12\x17\n\x0c\x61spect_ratio\x18\x02 \x01(\x02:\x01\x31\x12\x1d\n\x15min_padded_size_ratio\x18\x03 \x03(\x02\x12\x1d\n\x15max_padded_size_ratio\x18\x04 \x03(\x02\"5\n\x1b\x43onvertClassLogitsToSoftmax\x12\x16\n\x0btemperature\x18\x01 \x01(\x02:\x01\x31\"m\n\x15RandomSelfConcatImage\x12(\n\x1b\x63oncat_vertical_probability\x18\x01 \x01(\x02:\x03\x30.1\x12*\n\x1d\x63oncat_horizontal_probability\x18\x02 \x01(\x02:\x03\x30.1\"+\n\x10\x41utoAugmentImage\x12\x17\n\x0bpolicy_name\x18\x01 \x01(\t:\x02v0\"H\n\x1a\x44ropLabelProbabilistically\x12\r\n\x05label\x18\x01 \x01(\x05\x12\x1b\n\x10\x64rop_probability\x18\x02 \x01(\x02:\x01\x31\"9\n\x0bRemapLabels\x12\x17\n\x0foriginal_labels\x18\x01 \x03(\x05\x12\x11\n\tnew_label\x18\x02 \x01(\x05\"g\n\x11RandomJpegQuality\x12\x16\n\x0brandom_coef\x18\x01 \x01(\x02:\x01\x30\x12\x1b\n\x10min_jpeg_quality\x18\x02 \x01(\x05:\x01\x30\x12\x1d\n\x10max_jpeg_quality\x18\x03 \x01(\x05:\x03\x31\x30\x30\"}\n\x1dRandomDownscaleToTargetPixels\x12\x16\n\x0brandom_coef\x18\x01 \x01(\x02:\x01\x30\x12!\n\x11min_target_pixels\x18\x02 \x01(\x05:\x06\x33\x30\x30\x30\x30\x30\x12!\n\x11max_target_pixels\x18\x03 \x01(\x05:\x06\x35\x30\x30\x30\x30\x30\"\xa5\x01\n\x13RandomPatchGaussian\x12\x16\n\x0brandom_coef\x18\x01 \x01(\x02:\x01\x30\x12\x19\n\x0emin_patch_size\x18\x02 \x01(\x05:\x01\x31\x12\x1b\n\x0emax_patch_size\x18\x03 \x01(\x05:\x03\x32\x35\x30\x12\x1e\n\x13min_gaussian_stddev\x18\x04 \x01(\x02:\x01\x30\x12\x1e\n\x13max_gaussian_stddev\x18\x05 \x01(\x02:\x01\x31\"y\n\x17RandomSquareCropByScale\x12\x17\n\nmax_border\x18\x01 \x01(\x05:\x03\x31\x32\x38\x12\x16\n\tscale_min\x18\x02 \x01(\x02:\x03\x30.6\x12\x16\n\tscale_max\x18\x03 \x01(\x02:\x03\x31.3\x12\x15\n\nnum_scales\x18\x04 \x01(\x05:\x01\x38\"g\n\x1dRandomScaleCropAndPadToSquare\x12\x18\n\x0boutput_size\x18\x01 \x01(\x05:\x03\x35\x31\x32\x12\x16\n\tscale_min\x18\x02 \x01(\x02:\x03\x30.1\x12\x14\n\tscale_max\x18\x03 \x01(\x02:\x01\x32\"0\n\x0b\x41\x64justGamma\x12\x10\n\x05gamma\x18\x01 \x01(\x02:\x01\x31\x12\x0f\n\x04gain\x18\x02 \x01(\x02:\x01\x31')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.preprocessor_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PREPROCESSINGSTEP._serialized_start=72
  _PREPROCESSINGSTEP._serialized_end=3229
  _NORMALIZEIMAGE._serialized_start=3231
  _NORMALIZEIMAGE._serialized_end=3349
  _RANDOMHORIZONTALFLIP._serialized_start=3351
  _RANDOMHORIZONTALFLIP._serialized_end=3434
  _RANDOMVERTICALFLIP._serialized_start=3436
  _RANDOMVERTICALFLIP._serialized_end=3517
  _RANDOMROTATION90._serialized_start=3519
  _RANDOMROTATION90._serialized_end=3597
  _RANDOMPIXELVALUESCALE._serialized_start=3599
  _RANDOMPIXELVALUESCALE._serialized_end=3664
  _RANDOMIMAGESCALE._serialized_start=3666
  _RANDOMIMAGESCALE._serialized_end=3742
  _RANDOMRGBTOGRAY._serialized_start=3744
  _RANDOMRGBTOGRAY._serialized_end=3787
  _RANDOMADJUSTBRIGHTNESS._serialized_start=3789
  _RANDOMADJUSTBRIGHTNESS._serialized_end=3837
  _RANDOMADJUSTCONTRAST._serialized_start=3839
  _RANDOMADJUSTCONTRAST._serialized_end=3910
  _RANDOMADJUSTHUE._serialized_start=3912
  _RANDOMADJUSTHUE._serialized_end=3954
  _RANDOMADJUSTSATURATION._serialized_start=3956
  _RANDOMADJUSTSATURATION._serialized_end=4029
  _RANDOMDISTORTCOLOR._serialized_start=4031
  _RANDOMDISTORTCOLOR._serialized_end=4075
  _RANDOMJITTERBOXES._serialized_start=4077
  _RANDOMJITTERBOXES._serialized_end=4117
  _RANDOMCROPIMAGE._serialized_start=4120
  _RANDOMCROPIMAGE._serialized_end=4355
  _RANDOMPADIMAGE._serialized_start=4358
  _RANDOMPADIMAGE._serialized_end=4495
  _RANDOMABSOLUTEPADIMAGE._serialized_start=4497
  _RANDOMABSOLUTEPADIMAGE._serialized_end=4595
  _RANDOMCROPPADIMAGE._serialized_start=4598
  _RANDOMCROPPADIMAGE._serialized_end=4917
  _RANDOMCROPTOASPECTRATIO._serialized_start=4919
  _RANDOMCROPTOASPECTRATIO._serialized_end=5024
  _RANDOMBLACKPATCHES._serialized_start=5026
  _RANDOMBLACKPATCHES._serialized_end=5137
  _RANDOMRESIZEMETHOD._serialized_start=5139
  _RANDOMRESIZEMETHOD._serialized_end=5204
  _RGBTOGRAY._serialized_start=5206
  _RGBTOGRAY._serialized_end=5217
  _SCALEBOXESTOPIXELCOORDINATES._serialized_start=5219
  _SCALEBOXESTOPIXELCOORDINATES._serialized_end=5249
  _RESIZEIMAGE._serialized_start=5252
  _RESIZEIMAGE._serialized_end=5444
  _RESIZEIMAGE_METHOD._serialized_start=5377
  _RESIZEIMAGE_METHOD._serialized_end=5444
  _SUBTRACTCHANNELMEAN._serialized_start=5446
  _SUBTRACTCHANNELMEAN._serialized_end=5482
  _SSDRANDOMCROPOPERATION._serialized_start=5485
  _SSDRANDOMCROPOPERATION._serialized_end=5696
  _SSDRANDOMCROP._serialized_start=5698
  _SSDRANDOMCROP._serialized_end=5782
  _SSDRANDOMCROPPADOPERATION._serialized_start=5785
  _SSDRANDOMCROPPADOPERATION._serialized_end=6124
  _SSDRANDOMCROPPAD._serialized_start=6126
  _SSDRANDOMCROPPAD._serialized_end=6216
  _SSDRANDOMCROPFIXEDASPECTRATIOOPERATION._serialized_start=6219
  _SSDRANDOMCROPFIXEDASPECTRATIOOPERATION._serialized_end=6394
  _SSDRANDOMCROPFIXEDASPECTRATIO._serialized_start=6397
  _SSDRANDOMCROPFIXEDASPECTRATIO._serialized_end=6538
  _SSDRANDOMCROPPADFIXEDASPECTRATIOOPERATION._serialized_start=6541
  _SSDRANDOMCROPPADFIXEDASPECTRATIOOPERATION._serialized_end=6771
  _SSDRANDOMCROPPADFIXEDASPECTRATIO._serialized_start=6774
  _SSDRANDOMCROPPADFIXEDASPECTRATIO._serialized_end=6983
  _CONVERTCLASSLOGITSTOSOFTMAX._serialized_start=6985
  _CONVERTCLASSLOGITSTOSOFTMAX._serialized_end=7038
  _RANDOMSELFCONCATIMAGE._serialized_start=7040
  _RANDOMSELFCONCATIMAGE._serialized_end=7149
  _AUTOAUGMENTIMAGE._serialized_start=7151
  _AUTOAUGMENTIMAGE._serialized_end=7194
  _DROPLABELPROBABILISTICALLY._serialized_start=7196
  _DROPLABELPROBABILISTICALLY._serialized_end=7268
  _REMAPLABELS._serialized_start=7270
  _REMAPLABELS._serialized_end=7327
  _RANDOMJPEGQUALITY._serialized_start=7329
  _RANDOMJPEGQUALITY._serialized_end=7432
  _RANDOMDOWNSCALETOTARGETPIXELS._serialized_start=7434
  _RANDOMDOWNSCALETOTARGETPIXELS._serialized_end=7559
  _RANDOMPATCHGAUSSIAN._serialized_start=7562
  _RANDOMPATCHGAUSSIAN._serialized_end=7727
  _RANDOMSQUARECROPBYSCALE._serialized_start=7729
  _RANDOMSQUARECROPBYSCALE._serialized_end=7850
  _RANDOMSCALECROPANDPADTOSQUARE._serialized_start=7852
  _RANDOMSCALECROPANDPADTOSQUARE._serialized_end=7955
  _ADJUSTGAMMA._serialized_start=7957
  _ADJUSTGAMMA._serialized_end=8005
# @@protoc_insertion_point(module_scope)
