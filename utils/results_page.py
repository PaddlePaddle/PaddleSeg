"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import glob
import os
import numpy as np

id2cat = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic_light',
    7: 'traffic_sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'}

# Leaderboard mapillary
sota_iu_results = {
    0: 98.4046,
    1: 85.0224,
    2: 93.6462,
    3: 61.7487,
    4: 63.8885,
    5: 67.6745,
    6: 77.43,
    7: 80.8351,
    8: 93.7341,
    9: 71.8774,
    10: 95.6122,
    11: 86.7228,
    12: 72.7778,
    13: 95.7033,
    14: 79.9019,
    15: 93.0954,
    16: 89.7196,
    17: 72.5731,
    18: 78.2172,
    255: 0}


class ResultsPage(object):
    '''
    This creates an HTML page of embedded images, useful for showing evaluation results.

    Usage:
    ip = ImagePage(html_fn)

    # Add a table with N images ...
    ip.add_table((img, descr), (img, descr), ...)

    # Generate html page
    ip.write_page()
    '''

    def __init__(self, experiment_name, html_filename):
        self.experiment_name = experiment_name
        self.html_filename = html_filename
        self.outfile = open(self.html_filename, 'w')
        self.items = []

    def _print_header(self):
        header = '''<!DOCTYPE html>
<html>
  <head>
    <title>Experiment = {}</title>
  </head>
  <body>'''.format(self.experiment_name)
        self.outfile.write(header)

    def _print_footer(self):
        self.outfile.write('''  </body>
</html>''')

    def _print_table_header(self, table_name):
        table_hdr = '''    <h3>{}</h3>
    <table border="1" style="table-layout: fixed;">
      <tr>'''.format(table_name)
        self.outfile.write(table_hdr)

    def _print_table_footer(self):
        table_ftr = '''      </tr>
    </table>'''
        self.outfile.write(table_ftr)

    def _print_table_guts(self, img_fn, descr):
        table = '''        <td halign="center" style="word-wrap: break-word;" valign="top">
          <p>
            <a href="{img_fn}">
              <img src="{img_fn}" style="width:768px">
            </a><br>
            <p>{descr}</p>
          </p>
        </td>'''.format(img_fn=img_fn, descr=descr)
        self.outfile.write(table)

    def add_table(self, img_label_pairs, table_heading=''):
        """
        :img_label_pairs: A list of pairs of [img,label]
        """
        self.items.append([img_label_pairs, table_heading])

    def _write_table(self, table, heading):
        img, _descr = table[0]
        self._print_table_header(heading)
        for img, descr in table:
            self._print_table_guts(img, descr)
        self._print_table_footer()

    def write_page(self):
        self._print_header()

        for table, heading in self.items:
            self._write_table(table, heading)

        self._print_footer()

    def _print_page_start(self):
        page_start = '''<!DOCTYPE html>
<html>
<head>
<title>Experiment = EXP_NAME </title>
<style>
table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
}
th, td {
    padding: 5px;
    text-align: left;
}
</style>
</head>
<body>'''
        self.outfile.write(page_start)

    def _print_table_start(self, caption, hdr):
        self.outfile.write('''<table style="width:100%">
  <caption>{}</caption>
  <tr>'''.format(caption))
        for hdr_col in hdr:
            self.outfile.write('    <th>{}</th>'.format(hdr_col))
        self.outfile.write('  </tr>')

    def _print_table_row(self, row):
        self.outfile.write('  <tr>')
        for i in row:
            self.outfile.write('    <td>{}</td>'.format(i))
        # Create Links
        fp_link = '<a href="{}_fp.html">false positive Top N</a>'.format(row[
                                                                         1])
        fn_link = '<a href="{}_fn.html">false_negative Top N</a>'.format(row[
                                                                         1])
        self.outfile.write('    <td>{}</td>'.format(fp_link))
        self.outfile.write('    <td>{}</td>'.format(fn_link))
        self.outfile.write('  </tr>')

    def _print_table_end(self):
        self.outfile.write('</table>')

    def _print_page_end(self):
        self.outfile.write('''
</body>
</html>''')

    def create_main(self, iu, hist):
        self._print_page_start()
        #_print_table_style()
        # Calculate all of the terms:
        iu_false_positive = hist.sum(axis=1) - np.diag(hist)
        iu_false_negative = hist.sum(axis=0) - np.diag(hist)
        iu_true_positive = np.diag(hist)

        hdr = ("Class ID", "Class", "IoU", "Sota-IU", "TP",
               "FP", "FN", "precision", "recall", "", "")
        self._print_table_start("Mean IoU Results", hdr)
        for iu_score, index in iu:
            class_name = id2cat[index]
            iu_string = '{:5.2f}'.format(iu_score * 100)
            total_pixels = hist.sum()
            tp = '{:5.2f}'.format(100 * iu_true_positive[index] / total_pixels)
            fp = '{:5.2f}'.format(
                iu_false_positive[index] / iu_true_positive[index])
            fn = '{:5.2f}'.format(
                iu_false_negative[index] / iu_true_positive[index])
            precision = '{:5.2f}'.format(
                iu_true_positive[index] / (iu_true_positive[index] + iu_false_positive[index]))
            recall = '{:5.2f}'.format(
                iu_true_positive[index] / (iu_true_positive[index] + iu_false_negative[index]))
            sota = '{:5.2f}'.format(sota_iu_results[index])
            row = (index, class_name, iu_string, sota,
                   tp, fp, fn, precision, recall)
            self._print_table_row(row)
        self._print_table_end()
        self._print_page_end()


def main():
    images = glob.glob('dump_imgs_train/*.png')
    images = [i for i in images if 'mask' not in i]

    ip = ResultsPage('test page', 'dd.html')
    for img in images:
        basename = os.path.splitext(img)[0]
        mask_img = basename + '_mask.png'
        ip.add_table(((img, 'image'), (mask_img, 'mask')))
    ip.write_page()
