import hashlib
from .bencoding import Decoder

class Torrent:
    """
    Represents the data from a .torrent file.
    """
    def __init__(self, filename: str):
        # TODO:
        # 1. Read the .torrent file (in binary mode).
        # 2. Decode the bencoded data using the Decoder class.
        # 3. Store the decoded dictionary in `self.meta_info`.
        # 4. Extract tracker URL, piece length, etc., from the meta_info.
        # 5. Calculate the info_hash.
        self.meta_info = None
        self.info_hash = None
        self.announce_url = None
        self.piece_length = None
        self.piece_hashes = None
        self.length = None # Total size of the file(s)

    def _calculate_info_hash(self) -> bytes:
        """
        Calculates the SHA1 hash of the 'info' dictionary from the
        metainfo file.
        """
        # TODO:
        # 1. Bencode the self.meta_info['info'] dictionary.
        # 2. Return the SHA1 hash of the result.
        pass
