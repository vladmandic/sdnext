import base64
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.exceptions import HTTPException
from PIL import Image

from modules.api import endpoints, helpers, models
from modules.image import watermark


class TestAPIWatermark(unittest.TestCase):
    def test_mask_runs_for_rgb_watermark_received_by_api(self):
        def encode(image):
            with io.BytesIO() as buffer:
                image.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("ascii")

        self.addCleanup(watermark.load_watermark.cache_clear)
        watermark.load_watermark.cache_clear()
        with patch("modules.rembg.lucida.remove", return_value=Image.new("RGBA", (2, 2), "white")) as remove:
            watermark.post_watermark(
                image=encode(Image.new("RGB", (8, 8), "black")),
                wm_image=encode(Image.new("RGB", (2, 2), "white")),
                position="center",
                mask=True,
            )
        remove.assert_called_once()
        watermark.load_watermark.cache_clear()

    def test_watermark_decode_preserves_source_alpha_mode(self):
        image = Image.new("RGB", (2, 2), "white")
        encoded = watermark.encode_pil_to_base64(image, ext="PNG").decode("ascii")

        decoded = helpers.decode_base64_to_image(encoded, preserve_mode=True)

        self.assertEqual(decoded.mode, "RGB")

    def test_watermark_helper_applies_image_options_and_verifies_text(self):
        with patch.object(watermark, "decode_base64_to_image", side_effect=["image", "watermark"]), \
                patch.object(watermark, "set_image_watermark", return_value="visible") as apply_image, \
                patch.object(watermark, "set_text_watermark", return_value="complete") as apply_text, \
                patch.object(watermark, "encode_pil_to_base64", return_value="encoded") as encode:
            result = watermark.post_watermark(
                image="image-data",
                wm_text="hidden",
                wm_image="watermark-data",
                position="bottom/right",
                fmt="WEBP",
                quiet=True,
                mask=True,
                scale=0.5,
                opacity=0.7,
                pad=12,
                verify=True,
            )

        self.assertEqual(result, "encoded")
        self.assertEqual(apply_image.call_args.args, ("image", "watermark", "bottom/right"))
        self.assertEqual(apply_image.call_args.kwargs, {
            "mask": True,
            "scale": 0.5,
            "opacity": 0.7,
            "pad": 12,
            "quiet": True,
        })
        apply_text.assert_called_once_with("visible", "hidden", quiet=True, verify=True)
        encode.assert_called_once_with("complete", ext="WEBP")

    def test_post_image_forwards_optional_watermark_settings(self):
        request = models.ReqWatermark(
            image="image-data",
            wm_text="hidden",
            wm_image="watermark-data",
            position="bottom/right",
            mask=True,
            scale=0.5,
            opacity=0.7,
            pad=12,
            verify=True,
            image_format="WEBP",
        )
        with patch.object(watermark, "post_watermark", return_value="result") as apply:
            result = endpoints.post_watermark(request)

        self.assertEqual(result, "result")
        self.assertEqual(apply.call_args.kwargs, {
            "image": "image-data",
            "video": None,
            "video_format": None,
            "wm_text": "hidden",
            "wm_image": "watermark-data",
            "position": "bottom/right",
            "fmt": "WEBP",
            "mask": True,
            "scale": 0.5,
            "opacity": 0.7,
            "pad": 12,
            "verify": True,
            "quiet": True,
        })

    def test_post_requires_exactly_one_media_input(self):
        for request in (
            models.ReqWatermark(),
            models.ReqWatermark(image="image-data", video="video-data"),
        ):
            with self.subTest(request=request), self.assertRaises(HTTPException) as caught:
                endpoints.post_watermark(request)
            self.assertEqual(caught.exception.status_code, 400)

    def test_post_video_returns_encoded_result_and_forwards_options(self):
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, "watermarked.mp4")

            def fake_watermark(**kwargs):
                with open(output_path, "wb") as file:
                    file.write(b"watermarked video")
                self.assertEqual(kwargs["position"], "center")
                self.assertTrue(kwargs["mask"])
                self.assertEqual(kwargs["scale"], 0.75)
                self.assertEqual(kwargs["opacity"], 0.5)
                self.assertEqual(kwargs["pad"], 8)
                return output_path

            request = models.ReqWatermark(
                video=base64.b64encode(b"source video").decode("ascii"),
                video_format="mp4",
                wm_image="",
                position="center",
                mask=True,
                scale=0.75,
                opacity=0.5,
                pad=8,
            )
            with patch.object(watermark, "set_video_watermark", side_effect=fake_watermark):
                result = endpoints.post_watermark(request)

            self.assertEqual(base64.b64decode(result), b"watermarked video")
            self.assertFalse(os.path.exists(output_path))

    def test_video_rejects_image_only_text_watermarking(self):
        request = models.ReqWatermark(video="video-data", wm_text="hidden")
        with self.assertRaises(HTTPException) as caught:
            endpoints.post_watermark(request)
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
