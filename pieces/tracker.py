import aiohttp
import random
import urllib.parse
import struct

class Tracker:
    """
    Manages communication with a BitTorrent tracker.
    """
    def __init__(self, torrent):
        self.torrent = torrent
        # TODO: Generate a unique peer ID. A common practice is to use a
        # 20-byte string, e.g., '-PC0001-' followed by 12 random digits.
        self.peer_id = b'-PY0001-' + ''.join(str(random.randint(0, 9)) for _ in range(12)).encode()

    async def connect(self, first: bool = False, uploaded: int = 0, downloaded: int = 0):
        """
        Makes an announce request to the tracker.

        Args:
            first: Whether this is the first announce request.
            uploaded: The total number of bytes uploaded.
            downloaded: The total number of bytes downloaded.

        Returns:
            A list of (ip, port) tuples for the peers.
        """
        # TODO:
        # 1. Build the tracker GET request URL with all required parameters.
        # 2. Use aiohttp to make the request.
        # 3. Check for a 200 OK response.
        # 4. Decode the bencoded response.
        # 5. Parse the peers from the response.
        pass

    def _build_url(self, uploaded, downloaded, left) -> str:
        """Builds the URL for the tracker GET request."""
        # TODO: Implement the URL building logic.
        pass

    def _parse_peers(self, peers_bytes: bytes) -> list:
        """
        Parses the compact peer list from the tracker response.
        The peer list is a string of 6-byte chunks. Each chunk represents
        a peer, with the first 4 bytes being the IP address and the last
        2 bytes being the port number, both in network byte order.
        """
        # TODO: Implement the peer parsing logic.
        pass
