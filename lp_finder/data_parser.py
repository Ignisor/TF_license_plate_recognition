from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs
import cv2
import requests


class MongoLicensePlates(object):
    SECRET_KEY = 'sk_5964d3a17a85e259f901c72e'

    def __init__(self):
        self.client = MongoClient()
        self.db = self.client['lpr']
        self.collection = self.db['lp_images']
        self.fs = gridfs.GridFS(self.db)

    def insert(self, image, test):
        """
        Inserts image in DB
        :param image: binary image
        :param test: is used for test set 
        :return: id of record
        """
        file_id = self.fs.put(image)
        data = {
            'file_id': file_id,
            'is_lp': None,
            'lp_number': None,
            'test': test,
        }

        id_ = self.collection.insert_one(data).inserted_id

        return id_

    def update_lp_status(self, id_, is_lp, lp_num=None):
        if type(id_) != ObjectId:
            id_ = ObjectId(id_)

        data = {
            'is_lp': is_lp,
            'lp_num': lp_num,
        }

        self.collection.update_one({'_id': id_}, {'$set': data})

    def get_all(self):
        return self.collection.find()

    def get_train(self):
        return self.collection.find({'test': False})

    def get_test(self):
        return self.collection.find({'test': True})

    def get_unknown(self):
        return self.collection.find({'is_lp': None})

    def get(self, id_):
        if type(id_) != ObjectId:
            id_ = ObjectId(id_)

        return self.collection.find_one({'_id': id_}, projection={'_id': False})

    def get_image(self, file_id):
        if type(file_id) != ObjectId:
            file_id = ObjectId(file_id)

        return self.fs.get(file_id).read()

    def check_image(self, image_id):
        """
        checks image using openalpr cloud
        :param image_id: id of image file in DB
        :return: str:license_plate or None
        """
        img = self.get_image(image_id)
        file = {'image': img}

        url = f'https://api.openalpr.com/v2/recognize?recognize_vehicle=1&country=eu&secret_key={self.SECRET_KEY}'

        r = requests.post(url, files=file)

        if not r:
            raise ConnectionError(f'Can\'t connect to OpenALPR {r.json()}')

        results = r.json().get('results')
        plate = results[0].get('plate') if results else None

        return plate

    def check_all(self, only_unknown=True):
        objects = self.get_unknown() if only_unknown else self.get_all()
        for obj in objects:
            result = self.check_image(obj['file_id'])
            self.update_lp_status(obj['_id'], bool(result), result)
            print(f'Checked {obj["_id"]}: {result}')


class VideoCutter(object):
    SEARCH_RECTANGLE = ((400, 500), (1000, 700))

    def __init__(self):
        self.collection = MongoLicensePlates()

    def process_video(self, video_file, frameskip=3):
        """
        Split video in to frames, cut rectangles, and send them to db
        :param str video_file: path to video 
        :param int frameskip: how many frames to skip
        """
        video = cv2.VideoCapture(video_file)

        c = 0
        while video.isOpened():
            ret, frame = video.read()

            c += 1
            if c >= frameskip:
                c = 0

                # crop frame
                s_rect = self.SEARCH_RECTANGLE
                img = frame[s_rect[0][1]:s_rect[1][1], s_rect[0][0]:s_rect[1][0]]

                inserted_id = self.collection.insert(cv2.imencode('.jpg', img)[1].tostring())

                print(f'Inserted {inserted_id}')
