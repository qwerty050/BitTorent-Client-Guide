import pytest
from pieces.torrent import Torrent
from pieces.tracker import Tracker
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_torrent():
    """Fixture to create a mock Torrent object."""
    torrent = MagicMock(spec=Torrent)
    torrent.announce_url = "http://test-tracker.com/announce"
    torrent.info_hash = b'\x12\x34\x56\x78\x90\xab\xcd\xef\x12\x34\x56\x78\x90\xab\xcd\xef\x12\x34\x56\x78'
    torrent.length = 10000
    return torrent

@pytest.mark.asyncio
async def test_tracker_connect(mock_torrent):
    """Test a successful tracker announce."""
    tracker = Tracker(mock_torrent)

    # Mock response from the tracker
    mock_response_data = {
        b'interval': 1800,
        b'peers': b'\x08\x08\x08\x08\x1a\xe1\x08\x08\x04\x04\x1a\xe2' # 8.8.8.8:6881, 8.8.4.4:6882
    }

    # We use a patch to mock the aiohttp session
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Configure the mock to behave like a real aiohttp response
        mock_response = MagicMock()
        mock_response.status = 200
        # The bencoding encoder is part of the test setup here
        from pieces.bencoding import Encoder
        mock_response.read.return_value = Encoder(mock_response_data).encode()

        # Make the mock_get context manager return our mock response
        mock_get.return_value.__aenter__.return_value = mock_response

        peers = await tracker.connect()

        assert len(peers) == 2
        assert peers[0] == ('8.8.8.8', 6881)
        assert peers[1] == ('8.8.4.4', 6882)

def test_parse_peers(mock_torrent):
    """Test the peer parsing logic directly."""
    tracker = Tracker(mock_torrent)
    peers_bytes = b'\x7f\x00\x00\x01\x1a\xe1\xc0\xa8\x01\x01\x1a\xe2' # 127.0.0.1:6881, 192.168.1.1:6882
    peers = tracker._parse_peers(peers_bytes)
    assert peers == [('127.0.0.1', 6881), ('192.168.1.1', 6882)]

def test_parse_peers_malformed(mock_torrent):
    """Test peer parsing with malformed data."""
    tracker = Tracker(mock_torrent)
    peers_bytes = b'\x7f\x00\x00\x01\x1a' # Incomplete peer data
    with pytest.raises(ValueError):
        tracker._parse_peers(peers_bytes)
