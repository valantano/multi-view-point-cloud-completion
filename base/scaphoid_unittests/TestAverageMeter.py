import os, sys

import unittest
import torch

# Add the main directory to sys.path such that submodules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, "..", '..'))
if main_dir not in sys.path:
    sys.path.append(main_dir)

from base.scaphoid_utils.AverageMeter import AverageMeter  # Adjust to your actual module path


class TestAverageMeter(unittest.TestCase):

    def test_initialization_with_list(self):
        meter = AverageMeter(['loss', 'accuracy'])
        self.assertEqual(meter.get_names(), ['loss', 'accuracy'])
        self.assertEqual(meter.val(), [0, 0])
        self.assertTrue(meter.initialized)

    def test_reset(self):
        meter = AverageMeter(['a', 'b'])
        meter.update([1.0, 2.0])
        meter.reset()
        self.assertEqual(meter.val(), [0, 0])
        self.assertEqual(meter.count(), [0, 0])

    def test_single_update(self):
        meter = AverageMeter(['loss'])
        meter.update(5.0)
        self.assertEqual(meter.val(), 5.0)
        self.assertEqual(meter.sum(), 5.0)
        self.assertEqual(meter.count(), 1)
        self.assertEqual(meter.avg(), 5.0)

    def test_list_update(self):
        meter = AverageMeter(['loss', 'acc'])
        meter.update([2.0, 4.0])
        self.assertEqual(meter.val(), [2.0, 4.0])
        self.assertEqual(meter.count(), [1, 1])
        self.assertEqual(meter.sum(), [2.0, 4.0])

    def test_multiple_updates(self):
        meter = AverageMeter(['x'])
        meter.update(2.0)
        meter.update(4.0)
        self.assertEqual(meter.count(), 2)
        self.assertEqual(meter.sum(), 6.0)
        self.assertEqual(meter.avg(), 3.0)
        self.assertAlmostEqual(meter.std(), 1.4142, places=4)

    def test_update_via_dict_scalar_and_1d_tensor(self):
        meter = AverageMeter(init_later=True)
        metric_dict = {
            'loss': torch.tensor([1.0, 2.0, 3.0]),
            'acc': torch.tensor(0.5)
        }
        meter.update_via_dict(metric_dict)
        self.assertEqual(meter.get_names(), ['loss', 'acc'])
        self.assertEqual(meter.count('loss'), 3)
        self.assertEqual(meter.count('acc'), 1)
        self.assertEqual(meter.sum('loss'), 6.0)
        self.assertEqual(meter.sum('acc'), 0.5)

    def test_avg_sum_val_count_std_by_name_and_index(self):
        meter = AverageMeter(['mse', 'psnr'])
        meter.update([4.0, 8.0])
        self.assertEqual(meter.avg(0), 4.0)
        self.assertEqual(meter.avg('psnr'), 8.0)
        self.assertEqual(meter.val('mse'), 4.0)
        self.assertEqual(meter.count(1), 1)
        self.assertEqual(meter.std('psnr'), 0.0)

        meter.update([2.0, 6.0])
        self.assertEqual(meter.avg(0), 3.0)
        self.assertEqual(meter.avg('psnr'), 7.0)
        self.assertEqual(meter.val('mse'), 2.0)
        self.assertEqual(meter.count(1), 2)
        self.assertAlmostEqual(meter.std('psnr'), torch.std(torch.tensor([6.0, 8.0]), unbiased=True).item(), places=6)
        self.assertAlmostEqual(meter.std(0), torch.std(torch.tensor([2.0, 4.0]), unbiased=True).item(), places=6)

    def test_invalid_index(self):
        meter = AverageMeter(['a'])
        with self.assertRaises(IndexError):
            meter.update_single(1.0, 2)

    def test_uninitialized_update(self):
        meter = AverageMeter(init_later=True)
        with self.assertRaises(ValueError):
            meter.update(1.0)

    def test_update_with_invalid_dict(self):
        meter = AverageMeter(['a'])
        with self.assertRaises(NotImplementedError):
            meter.update({'a': [1.0, 2.0]})

    def test_get_epoch_log_dict(self):
        meter = AverageMeter(['CDL1', 'other'])
        meter.update([0.01, 0.1])
        log = meter.get_epoch_log_dict(val=True)
        self.assertIn('Metric/CDL1', log)
        self.assertIn('Metric/other', log)
        self.assertAlmostEqual(log['Metric/CDL1'], 10.0)
        self.assertAlmostEqual(log['Metric/other'], 0.1)


if __name__ == '__main__':
    unittest.main()
