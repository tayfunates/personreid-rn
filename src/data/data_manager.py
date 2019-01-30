"""Create dataset"""
from market1501 import Market1501
from market1501_rn import Market1501 as MarketRN

__img_factory = {
	'market1501': Market1501,
	'market1501rn': MarketRN
#    'cuhk03': CUHK03,
#    'dukemtmcreid': DukeMTMCreID,
#    'msmt17': MSMT17,
}

#__vid_factory = {
#    'mars': Mars,
#    'ilidsvid': iLIDSVID,
#    'prid': PRID,
#    'dukemtmcvidreid': DukeMTMCVidReID,
#}

def get_names():
#	return __img_factory.keys() + __vid_factory.keys()
	return __img_factory.keys()

def init_img_dataset(name, **kwargs):
	if name not in __img_factory.keys():
		raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
	return __img_factory[name](**kwargs)

#def init_vid_dataset(name, **kwargs):
#	if name not in __vid_factory.keys():
#		raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __vid_factory.keys()))
#	return __vid_factory[name](**kwargs)
