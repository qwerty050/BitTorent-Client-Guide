import pytest
from pieces.protocol import Handshake #, KeepAlive, Choke, ... etc.

def test_handshake_encode_decode():
    """Test that a Handshake message can be encoded and decoded."""
    info_hash = b'12345678901234567890'
    peer_id = b'my-awesome-peer-id-0'

    original_handshake = Handshake(info_hash, peer_id)
    encoded_handshake = original_handshake.encode()
    decoded_handshake = Handshake.decode(encoded_handshake)

    assert decoded_handshake.info_hash == info_hash
    assert decoded_handshake.peer_id == peer_id

# TODO: Add tests for all other message types once you define them.
# Example for a 'Request' message:
#
# def test_request_encode_decode():
#     original_request = Request(index=1, begin=32768, length=16384)
#     encoded = original_request.encode()
#     decoded = Request.decode(encoded)
#     assert decoded.index == 1
#     assert decoded.begin == 32768
#     assert decoded.length == 16384
