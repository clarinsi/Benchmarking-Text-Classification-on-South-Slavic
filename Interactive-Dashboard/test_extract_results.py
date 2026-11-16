import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import extract_results as extract


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def matrix(self, records):
        path = self.root / 'matrix.json'
        path.write_text(json.dumps(records), encoding='utf-8')
        return path

    def test_all_scripts_zero_scores_and_identical_duplicates(self):
        record = {'Model': 'openai/gpt-4o', 'sr_latn': 0.9, 'sr_cyrl': 0.8,
                  'sr-tor_latn': 0.7, 'sr-tor_cyrl': 0.6,
                  'sr-satra_latn': 0.5, 'sr-satra_cyrl': 0, 'en': None}
        rows = extract.extract_matrix_rows('PIQA', self.matrix([record, record]))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r['language']: r['value'] for r in rows}, {
            'Serbian': 0.9, 'Serbian (cyrillic)': 0.8,
            'Torlak Dialect': 0.7, 'Torlak Dialect (cyrillic)': 0.6,
            'Šatrovački Dialect': 0.5, 'Šatrovački Dialect (cyrillic)': 0.0})
        self.assertEqual({r['source_model'] for r in rows}, {'openai/gpt-4o'})

    def test_conflicting_duplicate_scores_fail(self):
        path = self.matrix([{'Model': 'openai/gpt-4o', 'en': 0.9},
                            {'Model': 'gpt-4o-2024-08-06', 'en': 0.8}])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            extract.extract_matrix_rows('COPA', path)

    def test_invalid_scores_and_unknown_languages_fail(self):
        for value in [float('nan'), float('inf'), -0.1, 1.1, True, 'bad']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                extract.extract_matrix_rows('PIQA', self.matrix([{'Model': 'm', 'en': value}]))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            extract.extract_matrix_rows('PIQA', self.matrix([{'Model': 'm', 'new-language': 0.8}]))

    def test_legacy_baselines_scripts_and_distinct_variants_survive(self):
        path = self.root / 'paper.json'
        records = [
            {'Model': 'dummy-most_frequent', 'Serbian (cyrillic) (accuracy)': 0.5},
            {'Model': 'deepseek-r1:14b', 'English (accuracy)': 0.6},
            {'Model': 'deepseek/deepseek-r1-0528', 'English (accuracy)': 0.9},
            {'Model': 'GaMS-27B-quantized', 'English (accuracy)': 0.7},
        ]
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        rows = extract.extract_rows_for_task('PIQA', path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r['model'] for r in rows}), 4)
        self.assertEqual({r['model'] for r in rows if r['source_model'].startswith('deepseek')},
                         {'DeepSeek-R1-Distill-Qwen-14B', 'DeepSeek-R1-0528'})

    def test_cli_uses_only_current_matrices_from_any_working_directory(self):
        # Stale models/cells must not leak into the current benchmark matrices.
        dash = self.root / 'repo/Interactive-Dashboard'
        dash.mkdir(parents=True)
        shutil.copy2(Path(extract.__file__), dash / 'extract_results.py')
        for task, source in extract.task_dict.items():
            if task in {'PIQA', 'COPA'}:
                continue
            relative = Path(source).resolve().relative_to(Path(extract.__file__).resolve().parent.parent)
            target = dash.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({'Model': 'gpt-4o-2024-08-06', 'English (accuracy)': 0.4}) + '\n')
        legacy_paths = []
        for bench, task in [('CLASSLA-PIQA-Benchmark', 'piqa'), ('DIALECT-COPA-Benchmark', 'copa')]:
            legacy = dash.parent / bench / 'evaluation-for-the-paper/results.json'
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text('\n'.join(json.dumps(r) for r in [
                {'Model': 'gpt-4o-2024-08-06', 'English (accuracy)': 0.4,
                 'Croatian (accuracy)': 0.3},
                {'Model': 'legacy-only-model', 'English (accuracy)': 0.2},
            ]) + '\n')
            legacy_paths.append(legacy)
            target = dash.parent / bench / 'results' / f'{task}_results.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps([{'Model': 'openai/gpt-4o', 'en': 0.9}]))
        result = subprocess.run([sys.executable, str(dash / 'extract_results.py')],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((dash / 'results.csv').exists(), result.stdout)
        with (dash / 'results.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r['task']: float(r['value']) for r in rows}, {
            'PIQA': 0.9, 'COPA': 0.9, 'Genre': 0.4, 'News Topic': 0.4,
            'Parliamentary Speech Topic': 0.4, 'Sentiment': 0.4})
        self.assertEqual({r['source_model'] for r in rows if r['task'] in {'PIQA', 'COPA'}},
                         {'openai/gpt-4o'})
        before = (dash / 'results.csv').read_bytes()
        for legacy in legacy_paths:
            legacy.unlink()
        result = subprocess.run([sys.executable, str(dash / 'extract_results.py')],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((dash / 'results.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
