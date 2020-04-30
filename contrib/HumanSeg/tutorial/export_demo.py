import HumanSeg

model = HumanSeg.models.load_model('output/best_model')

model.export_inference_model('output/export')
