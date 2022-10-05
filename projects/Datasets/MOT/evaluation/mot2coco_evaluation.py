import logging
from collections import OrderedDict, defaultdict
from pycocotools.coco import COCO


class COCOJSON(COCO):
    def __init__(self, annotations):
        """
        COCO by json variable, instead of json file
        """
        logger = logging.getLogger(__name__)

        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        logger.info('loading annotations into memory...')
        assert type(annotations) == dict, 'annotation file format {} not supported'.format(type(annotations))
        self.dataset = annotations
        self.createIndex()