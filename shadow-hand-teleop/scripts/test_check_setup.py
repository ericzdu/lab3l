"""Setup diagnostics must work without downloading model assets."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco

import check_setup


class SetupTests(unittest.TestCase):
    def run_check(self, args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = check_setup.main(args)
        return status, output.getvalue(), errors.getvalue()

    def test_default_checks_only_shadow_hand_and_arms_are_opt_in(self):
        with patch.object(check_setup, "check_model") as check:
            self.assertEqual(self.run_check([])[0], 0)
            self.assertEqual([call.args[0] for call in check.call_args_list], ["shadow_hand"])
            check.reset_mock()
            self.assertEqual(self.run_check(["--include-arms"])[0], 0)
            self.assertEqual([call.args[0] for call in check.call_args_list],
                             ["shadow_hand", "franka_emika_panda"])

    def test_missing_offline_assets_never_call_the_downloader(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = check_setup.menagerie.Cache(dir=directory)
            cache.root = None
            with patch.object(check_setup.menagerie, "Cache", return_value=cache), \
                    patch.object(check_setup.menagerie, "load") as load:
                status, output, errors = self.run_check(["--offline", "--include-arms"])
            self.assertEqual(status, 1)
            load.assert_not_called()
            self.assertIn("shadow_hand", errors)
            self.assertIn("franka_emika_panda", errors)
            self.assertIn("offline mode does not download", errors)
            self.assertIn("MENAGERIE_CACHE_DIR=", errors)
            self.assertIn("m.prefetch", errors)
            self.assertNotIn("Setup check passed", output)

    def test_offline_uses_environment_cache_and_simulates_cached_model(self):
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(check_setup.os.environ, {"MENAGERIE_CACHE_DIR": directory}), \
                patch.object(check_setup.menagerie.Cache, "is_cached", return_value=True), \
                patch.object(check_setup.menagerie, "load", return_value=model) as load:
            status, output, errors = self.run_check(["--offline"])
            self.assertEqual(status, 0)
            self.assertEqual(load.call_args.kwargs["cache"].dir, Path(directory))
            self.assertIn("Setup check passed", output)
            self.assertEqual(errors, "")

    def test_local_root_failure_has_root_specific_guidance(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(check_setup.os.environ, {"MENAGERIE_ROOT": directory}):
            status, _, errors = self.run_check(["--offline"])
            self.assertEqual(status, 1)
            self.assertIn("Check MENAGERIE_ROOT", errors)
            self.assertIn("shadow_hand", errors)
            self.assertNotIn("m.prefetch", errors)

    def test_download_failure_is_reported_without_a_traceback(self):
        with patch.object(check_setup.menagerie, "load",
                          side_effect=check_setup.menagerie.DownloadError("network unavailable")):
            status, _, errors = self.run_check([])
            self.assertEqual(status, 1)
            self.assertIn("shadow_hand: network unavailable", errors)
            self.assertIn("m.prefetch", errors)
            self.assertNotIn("Traceback", errors)


if __name__ == "__main__":
    unittest.main()
