import HumanSeg
from HumanSeg.utils import visualize

im_file = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly/pexel/img/person_detection__ds6/img/pexels-photo-704264.jpg'
model = HumanSeg.models.load_model('output/best_model')
result = model.predict(im_file)
visualize(im_file, result, save_dir='output/')
