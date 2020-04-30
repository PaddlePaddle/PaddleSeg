import HumanSeg

im_file = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly/images/8d308c9cc0326a3bdfc90f7f6e1813df89786122.jpg'
model = HumanSeg.models.load_model('output/best_model')
result = model.predict(im_file)
