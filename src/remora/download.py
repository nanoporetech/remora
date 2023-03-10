import os

import requests
from tqdm import tqdm

from remora import log, RemoraError

LOGGER = log.get_logger()


class ModelDownload:
    __url__ = "https://cdn.oxfordnanoportal.com/software/analysis/remora/"

    def __init__(self, path, force=False):
        self.path = path
        self.force = force

    def location(self, filename):
        return os.path.join(self.path, filename)

    def exists(self, filename):
        return os.path.exists(self.location(filename))

    def download(self, url_frag):
        filename = f"{url_frag}.pt"
        url = f"{self.__url__}{filename}"
        req = requests.get(url, stream=True)
        total = int(req.headers.get("content-length", 0))

        if self.exists(filename) and not self.force:
            LOGGER.info(
                f"Model already exists, skipping download of {filename}"
            )
            return

        with tqdm(
            total=total,
            unit="iB",
            ncols=100,
            unit_scale=True,
            disable=os.environ.get("LOG_SAFE", False),
        ) as t:
            try:
                with open(self.location(filename), "wb") as f:
                    for data in req.iter_content(1024):
                        f.write(data)
                        t.update(len(data))
            except OSError as e:
                raise RemoraError(
                    "File-system was unable to write download model. "
                    f"URL is {url} and intended download path was "
                    f"{self.location(filename)}.\nFull error message: {e}"
                )
        LOGGER.info(f"Model {filename} downloaded to {self.location(filename)}")
