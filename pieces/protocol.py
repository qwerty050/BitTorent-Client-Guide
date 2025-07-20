import struct

class Handshake:
    """
    A message to initiate a connection with a peer.
    Format: <pstrlen><pstr><reserved><info_hash><peer_id>
    """
    def __init__(self, info_hash: bytes, peer_id: bytes):
        # TODO: Initialize handshake attributes.
        pass

    def encode(self) -> bytes:
        """Encodes the handshake message to bytes."""
        # TODO: Implement the encoding logic.
        pass

    @classmethod
    def decode(cls, data: bytes):
        """Decodes bytes into a Handshake object."""
        # TODO: Implement the decoding logic.
        pass

# TODO: Define classes for other peer messages like:
# - KeepAlive
# - Choke (id=0)
# - Unchoke (id=1)
# - Interested (id=2)
# - NotInterested (id=3)
# - Have (id=4)
# - Bitfield (id=5)
# - Request (id=6)
# - Piece (id=7)
# - Cancel (id=8)
#
# Each class should have an `encode` and a `decode` method.
