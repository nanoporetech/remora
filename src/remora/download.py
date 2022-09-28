import os
import re
import requests
from tqdm import tqdm
from remora import log

LOGGER = log.get_logger()


class ModelDownload:

    __url__ = "https://nanoporetech.box.com/shared/static/"

    def __init__(self, path, force=False):
        self.path = path
        self.force = force

    def location(self, filename):
        return os.path.join(self.path, filename)

    def exists(self, filename):
        return os.path.exists(self.location(filename))

    def download(self, url_frag):
        url = f"{self.__url__}{url_frag}.pt"
        req = requests.get(url, stream=True)
        total = int(req.headers.get("content-length", 0))
        f_name = re.findall(
            'filename="([^"]+)', req.headers["content-disposition"]
        )[0]

        if self.exists(f_name) and not self.force:
            LOGGER.info(f"Model already exists, skipping download of {f_name}")
            return

        with tqdm(
            total=total,
            unit="iB",
            ascii=True,
            ncols=100,
            unit_scale=True,
            leave=False,
        ) as t:
            with open(self.location(f_name), "wb") as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        LOGGER.info(f"Model {f_name} downloaded to {self.location(f_name)}")
