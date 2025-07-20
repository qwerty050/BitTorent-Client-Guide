
# Building a BitTorrent Client in Python with asyncio: A Step-by-Step Expert Guide

## Introduction

The BitTorrent protocol represents a fundamental paradigm shift away from the centralized client-server model of data distribution. Instead of relying on a single, powerful server to deliver files to many clients, BitTorrent leverages the collective power of its users. In a BitTorrent "swarm," each participant, or "peer," who downloads a file simultaneously becomes an uploader, sharing the pieces they have already acquired with others. This decentralized architecture creates a system that is not only resilient to single points of failure but also becomes more efficient as its popularity grows—a stark contrast to a central server, which would buckle under the same load.2

At its heart, a BitTorrent client is a highly concurrent application. It must manage dozens, sometimes hundreds, of simultaneous network connections to other peers. These connections are typically slow, unreliable, and operate independently. This is a classic example of an I/O-bound problem, where the application spends most of its time waiting for network operations to complete rather than performing CPU-intensive calculations. For such problems, traditional multi-threaded programming can become a quagmire of locks, race conditions, and high memory overhead.

This is precisely the scenario where Python's `asyncio` framework excels. Introduced in Python 3.5, `asyncio` provides a robust infrastructure for writing single-threaded concurrent code using coroutines.4 By employing an event loop, an

`asyncio`-based application can efficiently manage a large number of I/O operations. When one operation (like reading data from a peer's socket) is waiting, the event loop can switch to another task that is ready to proceed, maximizing resource utilization without the complexity of threading.5

This guide provides an exhaustive, step-by-step walkthrough for building a simplified but functional BitTorrent client using Python and `asyncio`. The journey is divided into five core stages, designed to build understanding incrementally. We will begin by parsing the torrent's "metainfo" file to understand its contents. Next, we will communicate with a central tracker to discover other peers in the swarm. The third and most intensive stage involves implementing the peer wire protocol to exchange data directly with those peers. We will then construct the logic to manage the download process, verifying data integrity and saving it to disk. Finally, we will orchestrate all these components using `asyncio`'s powerful concurrency primitives. Each stage includes detailed explanations of the relevant protocol specifications, production-quality code, and comprehensive unit tests to ensure correctness and robustness.

## Stage 1: Parsing the Metainfo File - The Language of the Swarm

The first step for any BitTorrent client is to understand the task at hand. This information is contained within a `.torrent` file, officially known as a metainfo file.6 This file does not contain the data to be downloaded, but rather the metadata that describes it: the file's name, its size, a list of cryptographic hashes to verify its integrity, and the address of a "tracker" to find other peers. This metadata is encoded in a custom format called Bencoding.

### 1.1 The Bencoding Specification (BEP-3)

Bencoding (pronounced "B-encoding") is a simple and efficient data serialization format used to store and transmit loosely structured data throughout the BitTorrent ecosystem.4 It is a binary format that supports four fundamental data types 6:

-   **Byte Strings:** Encoded as `<length>:<string>`, where `<length>` is a base-10 integer representing the number of bytes in the string. For example, `4:spam` represents the 4-byte string `spam`. Note that these are byte strings, not Unicode strings, a critical distinction for implementation.4

-   **Integers:** Encoded as `i<number>e`. For example, `i123e` represents the integer 123, and `i-42e` represents -42. The specification forbids leading zeros for non-zero numbers (e.g., `i03e` is invalid) and disallows a negative zero (`i-0e`).6

-   **Lists:** Encoded as `l<bencoded_element_1><bencoded_element_2>...e`. A list can contain any combination of bencoded types. For example, `l4:spam4:eggsi123ee` decodes to a list containing two byte strings and an integer: `[b'spam', b'eggs', 123]`.4

-   **Dictionaries:** Encoded as `d<key_1><value_1><key_2><value_2>...e`. Keys must be byte strings. A crucial rule is that the keys must be sorted lexicographically based on their raw byte values before being encoded. This ensures that the bencoded representation of a given dictionary is always identical, which is vital for cryptographic hashing.6 For example,

    `d3:cow3:moo4:spam4:eggse` decodes to an ordered dictionary like `{b'cow': b'moo', b'spam': b'eggs'}`.


Understanding these rules is the prerequisite for building a compliant client.

### 1.2 Implementation: The Bencoding `Decoder` and `Encoder`

To work with bencoded data, we will implement two core components: a `Decoder` to parse byte strings into Python objects, and an `Encoder` to serialize Python objects back into bencoded byte strings. These will be housed in a `bencoding.py` module.

The `Decoder` is implemented as a recursive-descent parser. It inspects the first byte (the "token") to determine the data type and then calls a corresponding private method to handle the decoding of that type. The `Encoder` performs the reverse operation, checking the Python type of the input data and applying the correct bencoding rule.

Python

```
# bittorrent_client/bencoding.py

from collections import OrderedDict

TOKEN_INTEGER = b'i'
TOKEN_LIST = b'l'
TOKEN_DICT = b'd'
TOKEN_END = b'e'
TOKEN_STRING_SEPARATOR = b':'


class DecodeError(Exception):
    """Custom exception for bencoding decoding errors."""
    pass


class Encoder:
    """
    Encodes Python objects into bencoded byte strings.
    Supports int, str, bytes, list, and dict types.
    """
    def __init__(self, data):
        self._data = data

    def encode(self) -> bytes:
        """
        Public method to start the encoding process.
        """
        return self._encode_next(self._data)

    def _encode_next(self, data):
        if isinstance(data, int):
            return self._encode_integer(data)
        elif isinstance(data, str):
            return self._encode_string(data.encode('utf-8'))
        elif isinstance(data, bytes):
            return self._encode_string(data)
        elif isinstance(data, list):
            return self._encode_list(data)
        elif isinstance(data, dict):
            return self._encode_dict(data)
        else:
            raise TypeError(f"Unsupported type for encoding: {type(data)}")

    def _encode_integer(self, value: int) -> bytes:
        return TOKEN_INTEGER + str(value).encode('ascii') + TOKEN_END

    def _encode_string(self, value: bytes) -> bytes:
        return str(len(value)).encode('ascii') + TOKEN_STRING_SEPARATOR + value

    def _encode_list(self, data: list) -> bytes:
        encoded_list = b"".join(self._encode_next(item) for item in data)
        return TOKEN_LIST + encoded_list + TOKEN_END

    def _encode_dict(self, data: dict) -> bytes:
        # Keys must be sorted lexicographically as raw byte strings
        sorted_keys = sorted(data.keys())
        encoded_dict = b""
        for key in sorted_keys:
            # Ensure key is bytes for encoding
            encoded_key = key if isinstance(key, bytes) else key.encode('utf-8')
            encoded_dict += self._encode_string(encoded_key)
            encoded_dict += self._encode_next(data[key])
        return TOKEN_DICT + encoded_dict + TOKEN_END


class Decoder:
    """
    Decodes a bencoded sequence of bytes into Python objects.
    """
    def __init__(self, data: bytes):
        if not isinstance(data, bytes):
            raise TypeError('Argument "data" must be of type bytes')
        self._data = data
        self._index = 0

    def decode(self):
        """
        Public method to start the decoding process.
        """
        token = self._peek()
        if token is None:
            raise DecodeError("Unexpected end of data")

        if token == TOKEN_INTEGER:
            self._consume()
            return self._decode_integer()
        elif token == TOKEN_LIST:
            self._consume()
            return self._decode_list()
        elif token == TOKEN_DICT:
            self._consume()
            return self._decode_dict()
        elif token in b'0123456789':
            return self._decode_string()
        else:
            raise DecodeError(f"Invalid token at index {self._index}: {token}")

    def _peek(self):
        if self._index < len(self._data):
            return self._data[self._index:self._index + 1]
        return None

    def _consume(self):
        self._index += 1

    def _read(self, length: int) -> bytes:
        if self._index + length > len(self._data):
            raise DecodeError("Unexpected end of data while reading")
        start = self._index
        self._index += length
        return self._data[start:self._index]

    def _read_until(self, token: bytes) -> bytes:
        try:
            occurrence = self._data.index(token, self._index)
            result = self._data[self._index:occurrence]
            self._index = occurrence + 1
            return result
        except ValueError:
            raise DecodeError(f"Token '{token.decode()}' not found")

    def _decode_integer(self):
        encoded_int = self._read_until(TOKEN_END)
        str_int = encoded_int.decode('ascii')

        # Validate integer format
        if (str_int.startswith('0') and len(str_int) > 1) or str_int.startswith('-0'):
            raise DecodeError(f"Invalid integer format: {str_int}")

        return int(str_int)

    def _decode_string(self) -> bytes:
        len_str = self._read_until(TOKEN_STRING_SEPARATOR)
        try:
            length = int(len_str)
        except ValueError:
            raise DecodeError(f"Invalid string length: {len_str}")
        return self._read(length)

    def _decode_list(self) -> list:
        decoded_list =
        while self._peek()!= TOKEN_END:
            decoded_list.append(self.decode())
        self._consume()  # Consume the 'e'
        return decoded_list

    def _decode_dict(self) -> OrderedDict:
        decoded_dict = OrderedDict()
        while self._peek()!= TOKEN_END:
            key = self._decode_string()
            value = self.decode()
            decoded_dict[key] = value
        self._consume()  # Consume the 'e'
        return decoded_dict

```

### 1.3 Testing the Bencoding Module

Rigorous testing is essential to ensure our bencoding implementation is correct. A failure here would prevent the client from functioning at all. We will use the `pytest` framework to create a comprehensive test suite. The tests will cover normal cases, edge cases like empty containers, and error conditions for malformed data.

Python

```
# tests/test_bencoding.py

import pytest
from collections import OrderedDict
from bittorrent_client.bencoding import Encoder, Decoder, DecodeError

# --- Encoder Tests ---

def test_encode_integer():
    assert Encoder(42).encode() == b'i42e'
    assert Encoder(-42).encode() == b'i-42e'
    assert Encoder(0).encode() == b'i0e'

def test_encode_string():
    assert Encoder("spam").encode() == b'4:spam'
    assert Encoder("").encode() == b'0:'

def test_encode_bytes():
    assert Encoder(b"spam").encode() == b'4:spam'

def test_encode_list():
    data = ["spam", "eggs", 123]
    expected = b'l4:spam4:eggsi123ee'
    assert Encoder(data).encode() == expected

def test_encode_dict():
    # Keys must be sorted for encoding
    data = OrderedDict()
    data[b'spam'] = b'eggs'
    data[b'cow'] = b'moo'
    expected = b'd3:cow3:moo4:spam4:eggse'
    assert Encoder(data).encode() == expected

def test_encode_nested_structure():
    data = OrderedDict()])])
    ])
    expected = b'd5:filesld6:lengthi10e4:pathl1:a1:bee10:publisher3:bob17:publisher-webpage15:www.example.come'
    assert Encoder(data).encode() == expected

# --- Decoder Tests ---

def test_decode_integer():
    assert Decoder(b'i42e').decode() == 42
    assert Decoder(b'i-42e').decode() == -42
    assert Decoder(b'i0e').decode() == 0

def test_decode_invalid_integer():
    with pytest.raises(DecodeError, match="Invalid integer format: 03"):
        Decoder(b'i03e').decode()
    with pytest.raises(DecodeError, match="Invalid integer format: -0"):
        Decoder(b'i-0e').decode()

def test_decode_string():
    assert Decoder(b'4:spam').decode() == b'spam'
    assert Decoder(b'0:').decode() == b''

def test_decode_list():
    data = b'l4:spam4:eggsi123ee'
    expected = [b'spam', b'eggs', 123]
    assert Decoder(data).decode() == expected

def test_decode_dict():
    data = b'd3:cow3:moo4:spam4:eggse'
    expected = OrderedDict([(b'cow', b'moo'), (b'spam', b'eggs')])
    assert Decoder(data).decode() == expected

def test_decode_nested_structure():
    data = b'd5:filesld6:lengthi10e4:pathl1:a1:bee10:publisher3:bob17:publisher-webpage15:www.example.come'
    expected = OrderedDict()])]),
        (b'publisher', b'bob'),
        (b'publisher-webpage', b'www.example.com')
    ])
    assert Decoder(data).decode() == expected

def test_decode_malformed_data():
    with pytest.raises(DecodeError):
        Decoder(b'i42').decode()  # Missing 'e'
    with pytest.raises(DecodeError):
        Decoder(b'l4:spam').decode() # Missing 'e'
    with pytest.raises(DecodeError):
        Decoder(b'5:spam').decode() # Length mismatch

# --- Round-trip Tests ---

@pytest.mark.parametrize("data", [
    42,
    "hello world",
    b"binary data",
    ["a", 1, ["b", 2]],
    OrderedDict([(b'a', 1), (b'b', )])
])
def test_round_trip(data):
    encoded = Encoder(data).encode()
    decoded = Decoder(encoded).decode()

    # Python's dicts are unordered by default, so we need to handle this
    if isinstance(data, dict):
        assert decoded == data
    elif isinstance(data, str):
        assert decoded == data.encode('utf-8')
    else:
        assert decoded == data

```

### 1.4 Data Model: The `Torrent` Class

Parsing the bencoded data is only the first half of the problem. The raw Python objects, with their byte-string keys, are inconvenient to work with. We will create a `Torrent` data class to act as a clean, high-level interface to the metainfo. This class will take the raw decoded dictionary as input and expose the torrent's properties through a user-friendly API.

Most importantly, this class will be responsible for calculating the `info_hash`. The `info_hash` is the 20-byte SHA1 hash of the _bencoded value_ of the `info` dictionary within the metainfo file.5 This hash is the torrent's unique identifier, used in all communications with trackers and peers. A failure to calculate this hash correctly means the client will be unable to join the correct swarm.

The process is subtle: the `info` dictionary must first be encoded back into its bencoded byte-string representation, and only then can the SHA1 hash be applied. This is where the correctness of our `Encoder`, particularly its handling of sorted dictionary keys, becomes paramount. A different key order would produce a different byte string and thus a completely different `info_hash`, rendering the client unable to connect to the correct swarm.6 This causal chain—from sorted keys to correct bencoding to correct hash to successful communication—is the bedrock of the entire protocol.

Python

```
# bittorrent_client/torrent.py

import hashlib
from collections import OrderedDict
from bittorrent_client.bencoding import Encoder, Decoder

class Torrent:
    """
    Represents the data from a.torrent file.
    """
    def __init__(self, metainfo: dict):
        self._metainfo = metainfo

    @classmethod
    def from_file(cls, file_path: str):
        """
        Creates a Torrent object from a.torrent file.
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        metainfo = Decoder(data).decode()
        return cls(metainfo)

    @property
    def info_hash(self) -> bytes:
        """
        Calculates the SHA1 hash of the bencoded 'info' dictionary.
        """
        info_dict = self._metainfo[b'info']
        bencoded_info = Encoder(info_dict).encode()
        return hashlib.sha1(bencoded_info).digest()

    @property
    def announce_url(self) -> str:
        """
        Returns the announce URL of the tracker.
        """
        return self._metainfo[b'announce'].decode('utf-8')

    @property
    def announce_list(self) -> list:
        """
        Returns the announce-list for multi-tracker support.
        """
        return self._metainfo.get(b'announce-list',)

    @property
    def piece_hashes(self) -> list[bytes]:
        """
        Returns a list of the SHA1 hashes for each piece.
        """
        pieces_raw = self._metainfo[b'info'][b'pieces']
        # Each hash is 20 bytes long
        return [pieces_raw[i:i+20] for i in range(0, len(pieces_raw), 20)]

    @property
    def piece_length(self) -> int:
        """
        Returns the length of a single piece in bytes.
        """
        return self._metainfo[b'info'][b'piece length']

    @property
    def total_size(self) -> int:
        """
        Returns the total size of the file(s) in the torrent.
        Handles both single-file and multi-file torrents.
        """
        info = self._metainfo[b'info']
        if b'length' in info:
            # Single-file torrent
            return info[b'length']
        elif b'files' in info:
            # Multi-file torrent
            return sum(file[b'length'] for file in info[b'files'])
        else:
            raise ValueError("Torrent metainfo must contain 'length' or 'files'")

    @property
    def name(self) -> str:
        """
        Returns the suggested name for the torrent.
        """
        return self._metainfo[b'info'][b'name'].decode('utf-8')

```

## Stage 2: Communicating with the Tracker

Having parsed the `.torrent` file, the client knows _what_ to download. The next step is to find out _from whom_. This is the role of the BitTorrent tracker, a central server that maintains a list of peers participating in a swarm.2 Our client will make an HTTP request to the tracker's "announce" URL to register itself and receive a list of peers to connect to.

### 2.1 The Tracker Announce Protocol (BEP-3, BEP-23)

Communication with an HTTP tracker is performed via a simple HTTP GET request.10 The client sends a request to the

`announce` URL specified in the metainfo file, with several key-value pairs appended as URL query parameters. These parameters identify the torrent, the client, and the client's download status.9

While the original specification allowed for a verbose, dictionary-based list of peers in the response, the modern de facto standard is the "compact" format, defined in BEP-23.11 By setting the

`compact=1` parameter in its request, the client signals that it can understand this more efficient format. Many modern trackers will only respond to requests that include this parameter, making its implementation essential for a compatible client.9 The following table summarizes the essential parameters for our initial announce request.

Parameter

Description

Source

Example

`info_hash`

URL-encoded 20-byte SHA1 hash of the bencoded `info` dictionary.

BEP-3

`%124Vx%9A...`

`peer_id`

URL-encoded 20-byte unique ID for our client.

BEP-3

`-PC0100-123456789012`

`port`

The TCP port our client will listen on (e.g., 6881).

BEP-3

`6881`

`uploaded`

Total bytes uploaded so far (base 10 ASCII).

BEP-3

`0`

`downloaded`

Total bytes downloaded so far (base 10 ASCII).

BEP-3

`0`

`left`

Total bytes remaining to be downloaded (base 10 ASCII).

BEP-3

`1485881344`

`compact`

Set to `1` to indicate we accept the compact peer list format.

BEP-23

`1`

`event`

(Optional) `started`, `completed`, or `stopped`.

BEP-3

`started`

### 2.2 Implementation: The `Tracker` Class

We will create a `Tracker` class to encapsulate the logic for communicating with the tracker. Since this involves network I/O, we will use the `aiohttp` library to make asynchronous HTTP requests, preventing our client from blocking while waiting for the tracker's response.4 The class will have a primary

`async` method, `connect`, which constructs the request URL, sends it, and parses the response.

Python

```
# bittorrent_client/tracker.py

import asyncio
import random
import socket
import struct
from urllib.parse import urlencode
import aiohttp
from bittorrent_client.bencoding import Decoder
from bittorrent_client.torrent import Torrent

class TrackerError(Exception):
    """Custom exception for tracker-related errors."""
    pass

class TrackerResponse:
    """
    Represents a successful response from the tracker.
    """
    def __init__(self, response: dict):
        self._response = response

    @property
    def interval(self) -> int:
        """Interval in seconds that the client should wait before re-requesting."""
        return self._response.get(b'interval', 1800)

    @property
    def peers(self) -> list[tuple[str, int]]:
        """
        Parses the compact peer list from the tracker response.
        Each peer is a tuple of (ip, port).
        """
        peers_raw = self._response.get(b'peers', b'')
        if not isinstance(peers_raw, bytes):
            # Handle non-compact response (dictionary model), though less common
            # For this simplified client, we'll focus on the compact model
            return

        # Peers are sent as a string of 6-byte chunks (4 for IP, 2 for port)
        peers_list =
        for i in range(0, len(peers_raw), 6):
            chunk = peers_raw[i:i+6]
            if len(chunk) < 6:
                continue
            ip_bytes = chunk[:4]
            port_bytes = chunk[4:]
            ip_address = socket.inet_ntoa(ip_bytes)
            port = struct.unpack('>H', port_bytes) # Big-endian, unsigned short
            peers_list.append((ip_address, port))
        return peers_list


class Tracker:
    """
    Manages communication with the BitTorrent tracker.
    """
    def __init__(self, torrent: Torrent):
        self._torrent = torrent
        self._peer_id = self._generate_peer_id()
        self._http_client = aiohttp.ClientSession()

    def _generate_peer_id(self) -> bytes:
        """Generates a 20-byte peer ID (Azureus-style)."""
        client_id = b'-PC0100-' # 'PC' for Python Client, '0100' for version 1.0.0
        random_bytes = ''.join(str(random.randint(0, 9)) for _ in range(12)).encode('ascii')
        return client_id + random_bytes

    async def connect(self, first: bool, uploaded: int = 0, downloaded: int = 0) -> TrackerResponse:
        """
        Connects to the tracker and announces the client's presence.
        """
        params = {
            'info_hash': self._torrent.info_hash,
            'peer_id': self._peer_id,
            'port': 6881, # The port our client will listen on
            'uploaded': uploaded,
            'downloaded': downloaded,
            'left': self._torrent.total_size - downloaded,
            'compact': 1,
        }
        if first:
            params['event'] = 'started'

        url = self._torrent.announce_url + '?' + urlencode(params)

        try:
            async with self._http_client.get(url) as response:
                if not response.status == 200:
                    raise TrackerError(f"Tracker returned HTTP status {response.status}")

                data = await response.read()
                decoded_data = Decoder(data).decode()

                if b'failure reason' in decoded_data:
                    raise TrackerError(f"Tracker error: {decoded_data[b'failure reason'].decode('utf-8')}")

                return TrackerResponse(decoded_data)
        except aiohttp.ClientError as e:
            raise TrackerError(f"Failed to connect to tracker: {e}")

    async def close(self):
        """Closes the aiohttp client session."""
        await self._http_client.close()

    @property
    def peer_id(self) -> bytes:
        return self._peer_id

```

### 2.3 Handling the Tracker Response

The tracker's response is a bencoded dictionary. We use our `Decoder` from Stage 1 to parse it into a Python `OrderedDict`. A successful response will contain two primary keys 11:

-   `interval`: An integer specifying the number of seconds the client should wait before sending its next periodic announce request.

-   `peers`: The list of peers. In the compact format we requested, this is a single byte string. We must parse this string by reading it in 6-byte chunks. For each chunk, the first 4 bytes represent the peer's IP address and the next 2 bytes represent its port number, both in network byte order (big-endian).9


The `TrackerResponse` class above abstracts this parsing logic, providing clean properties to access the interval and the parsed list of peer tuples. If the tracker encounters an error, it will instead return a dictionary with a `failure reason` key, which our `Tracker.connect` method checks for and raises an exception accordingly.10

### 2.4 Testing the `Tracker` Class

To test the `Tracker` class without making actual network requests, we will use `pytest` along with a mocking library. The `unittest.mock` module (or `pytest-mock`) is suitable for this. We will mock the `aiohttp.ClientSession.get` method to return a controlled `AsyncMock` that simulates a tracker's response. This allows us to test our URL construction, response parsing, and error handling logic in isolation.

Python

```
# tests/test_tracker.py

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from bittorrent_client.torrent import Torrent
from bittorrent_client.tracker import Tracker, TrackerError, TrackerResponse
from bittorrent_client.bencoding import Encoder

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_torrent():
    """Fixture for a mock Torrent object."""
    metainfo = {
        b'announce': b'http://test.tracker/announce',
        b'info': {
            b'name': b'test.file',
            b'piece length': 256,
            b'pieces': b'\x00' * 20, # Dummy hash
            b'length': 1024
        }
    }
    torrent = Torrent(metainfo)
    return torrent

async def test_tracker_connect_success(mock_torrent):
    """Test a successful connection to the tracker."""
    tracker = Tracker(mock_torrent)

    # A valid compact peer response: one peer at 127.0.0.1:6889
    peer_ip = b'\x7f\x00\x00\x01' # 127.0.0.1
    peer_port = b'\x1a\xe9' # 6889

    response_data = {
        b'interval': 1800,
        b'peers': peer_ip + peer_port
    }
    encoded_response = Encoder(response_data).encode()

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.read.return_value = encoded_response

    with patch('aiohttp.ClientSession.get', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))) as mock_get:
        response = await tracker.connect(first=True)

        # Verify the URL was constructed correctly
        args, kwargs = mock_get.call_args
        url = args
        assert 'info_hash=' in url
        assert 'peer_id=' in url
        assert 'port=6881' in url
        assert 'event=started' in url
        assert 'compact=1' in url

        # Verify the response is parsed correctly
        assert isinstance(response, TrackerResponse)
        assert response.interval == 1800
        assert response.peers == [('127.0.0.1', 6889)]

    await tracker.close()

async def test_tracker_failure_reason(mock_torrent):
    """Test handling of a tracker response with a failure reason."""
    tracker = Tracker(mock_torrent)

    response_data = {
        b'failure reason': b'Invalid info_hash'
    }
    encoded_response = Encoder(response_data).encode()

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.read.return_value = encoded_response

    with patch('aiohttp.ClientSession.get', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))):
        with pytest.raises(TrackerError, match="Tracker error: Invalid info_hash"):
            await tracker.connect(first=True)

    await tracker.close()

async def test_tracker_http_error(mock_torrent):
    """Test handling of a non-200 HTTP status from the tracker."""
    tracker = Tracker(mock_torrent)

    mock_response = AsyncMock()
    mock_response.status = 404

    with patch('aiohttp.ClientSession.get', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))):
        with pytest.raises(TrackerError, match="Tracker returned HTTP status 404"):
            await tracker.connect(first=True)

    await tracker.close()

```

## Stage 3: Implementing the Peer Wire Protocol

After obtaining a list of peer addresses from the tracker, the client can initiate direct communication to exchange file data. This communication is governed by the peer wire protocol, a TCP-based, stateful protocol that forms the core of BitTorrent's data transfer mechanism.6

### 3.1 Protocol Fundamentals

All communication between two peers over a TCP connection follows a specific structure. After an initial handshake, peers exchange a stream of length-prefixed messages. The standard message format is `<length_prefix:4><message_id:1><payload:variable>`, where `length_prefix` is a 4-byte big-endian integer specifying the length of the message that follows (ID + payload), and `message_id` is a single byte identifying the message type.4

**The Handshake:** The very first message sent by the initiating peer must be the handshake. It has a unique, fixed-length structure and does not follow the standard length-prefix format. Its structure is: `<pstrlen:1><pstr:19><reserved:8><info_hash:20><peer_id:20>`.9

-   `pstrlen`: A single byte with the value 19.

-   `pstr`: The 19-byte string "BitTorrent protocol".

-   `reserved`: 8 bytes, all set to zero for our purposes.

-   `info_hash`: The 20-byte SHA1 hash for the torrent.

-   `peer_id`: The 20-byte ID of the sending peer.


Upon receiving a connection, a peer responds with its own handshake. Both peers must then verify that the `info_hash` in the received handshake matches their own. If it doesn't, the connection is immediately dropped, as the other peer is part of a different swarm.13

**The State Machine:** Peer interaction is governed by a simple but crucial state machine with four states 4:

-   `am_choking`: A boolean indicating if our client is "choking" the remote peer. If true, we will not respond to their data requests.

-   `am_interested`: A boolean indicating if our client is "interested" in pieces the remote peer has.

-   `peer_choking`: A boolean indicating if the remote peer is choking our client. If true, they will not respond to our data requests.

-   `peer_interested`: A boolean indicating if the remote peer is interested in pieces our client has.


Initially, both peers are choking each other and are not interested. To begin downloading, our client must send an `Interested` message. If the remote peer is willing to share, it will respond with an `Unchoke` message. Only then is our client permitted to send `Request` messages for data blocks. This "tit-for-tat" mechanism is the foundation of BitTorrent's fairness policy.3

The following table summarizes the core messages of the peer wire protocol that we will implement.

ID

Name

Payload Description

-

`KeepAlive`

None (length prefix is 0). Sent periodically to keep the connection open.

0

`Choke`

None. Informs the peer they are now choked.

1

`Unchoke`

None. Informs the peer they are now unchoked.

2

`Interested`

None. Informs the peer we are interested in their pieces.

3

`Not Interested`

None. Informs the peer we are no longer interested.

4

`Have`

4-byte piece index (big-endian). Announces a newly acquired piece.

5

`Bitfield`

Variable-length byte array where each bit represents a piece the peer has.

6

`Request`

4-byte piece index, 4-byte block offset, 4-byte block length (all big-endian).

7

`Piece`

4-byte piece index, 4-byte block offset (both big-endian), and the variable-length block data.

8

`Cancel`

Identical to `Request` payload. Cancels a pending request.

### 3.2 Implementation: Message Classes

To manage these messages cleanly, we will create a class for each type in a `protocol.py` module. Each class will be responsible for encoding itself into a byte string for sending and decoding a byte string into a new instance upon receipt. We will use Python's `struct` module for packing and unpacking binary data according to the specified byte order and sizes.4

Python

```
# bittorrent_client/protocol.py

import struct

# Standard block size is 2^14 bytes
BLOCK_SIZE = 16384

class PeerMessage:
    """Base class for all peer wire protocol messages."""
    def encode(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def decode(cls, data: bytes):
        raise NotImplementedError

class Handshake(PeerMessage):
    """<pstrlen><pstr><reserved><info_hash><peer_id>"""
    PROTOCOL_STRING = b'BitTorrent protocol'
    LENGTH = 49 + len(PROTOCOL_STRING)

    def __init__(self, info_hash: bytes, peer_id: bytes):
        if len(info_hash)!= 20 or len(peer_id)!= 20:
            raise ValueError("Handshake info_hash and peer_id must be 20 bytes")
        self.info_hash = info_hash
        self.peer_id = peer_id

    def encode(self) -> bytes:
        return struct.pack(
            '>B19s8x20s20s',
            len(self.PROTOCOL_STRING),
            self.PROTOCOL_STRING,
            self.info_hash,
            self.peer_id
        )

    @classmethod
    def decode(cls, data: bytes):
        if len(data) < cls.LENGTH:
            raise ValueError("Handshake data too short")

        pstrlen, pstr, info_hash, peer_id = struct.unpack(
            '>B19s8x20s20s',
            data
        )

        if pstrlen!= len(cls.PROTOCOL_STRING) or pstr!= cls.PROTOCOL_STRING:
            raise ValueError("Invalid BitTorrent protocol string")

        return cls(info_hash, peer_id)

    def __eq__(self, other):
        return (isinstance(other, Handshake) and
                self.info_hash == other.info_hash and
                self.peer_id == other.peer_id)

class KeepAlive(PeerMessage):
    """<len=0000>"""
    MESSAGE_ID = -1 # No ID for KeepAlive

    def encode(self) -> bytes:
        return struct.pack('>I', 0)

    def __eq__(self, other):
        return isinstance(other, KeepAlive)

class Choke(PeerMessage):
    """<len=0001><id=0>"""
    MESSAGE_ID = 0

    def encode(self) -> bytes:
        return struct.pack('>IB', 1, self.MESSAGE_ID)

    def __eq__(self, other):
        return isinstance(other, Choke)

class Unchoke(PeerMessage):
    """<len=0001><id=1>"""
    MESSAGE_ID = 1

    def encode(self) -> bytes:
        return struct.pack('>IB', 1, self.MESSAGE_ID)

    def __eq__(self, other):
        return isinstance(other, Unchoke)

class Interested(PeerMessage):
    """<len=0001><id=2>"""
    MESSAGE_ID = 2

    def encode(self) -> bytes:
        return struct.pack('>IB', 1, self.MESSAGE_ID)

    def __eq__(self, other):
        return isinstance(other, Interested)

class NotInterested(PeerMessage):
    """<len=0001><id=3>"""
    MESSAGE_ID = 3

    def encode(self) -> bytes:
        return struct.pack('>IB', 1, self.MESSAGE_ID)

    def __eq__(self, other):
        return isinstance(other, NotInterested)

class Have(PeerMessage):
    """<len=0005><id=4><piece_index>"""
    MESSAGE_ID = 4

    def __init__(self, piece_index: int):
        self.piece_index = piece_index

    def encode(self) -> bytes:
        return struct.pack('>IBI', 5, self.MESSAGE_ID, self.piece_index)

    @classmethod
    def decode(cls, data: bytes):
        _, piece_index = struct.unpack('>BI', data)
        return cls(piece_index)

    def __eq__(self, other):
        return (isinstance(other, Have) and
                self.piece_index == other.piece_index)

class Bitfield(PeerMessage):
    """<len=0001+X><id=5><bitfield>"""
    MESSAGE_ID = 5

    def __init__(self, bitfield: bytes):
        self.bitfield = bitfield

    def encode(self) -> bytes:
        length = 1 + len(self.bitfield)
        return struct.pack(f'>IB{len(self.bitfield)}s', length, self.MESSAGE_ID, self.bitfield)

    @classmethod
    def decode(cls, data: bytes):
        return cls(data[1:])

    def __eq__(self, other):
        return (isinstance(other, Bitfield) and
                self.bitfield == other.bitfield)

class Request(PeerMessage):
    """<len=0013><id=6><index><begin><length>"""
    MESSAGE_ID = 6

    def __init__(self, piece_index: int, begin: int, length: int = BLOCK_SIZE):
        self.piece_index = piece_index
        self.begin = begin
        self.length = length

    def encode(self) -> bytes:
        return struct.pack('>IBIII', 13, self.MESSAGE_ID, self.piece_index, self.begin, self.length)

    @classmethod
    def decode(cls, data: bytes):
        _, piece_index, begin, length = struct.unpack('>BIII', data)
        return cls(piece_index, begin, length)

    def __eq__(self, other):
        return (isinstance(other, Request) and
                self.piece_index == other.piece_index and
                self.begin == other.begin and
                self.length == other.length)

class Piece(PeerMessage):
    """<len=0009+X><id=7><index><begin><block>"""
    MESSAGE_ID = 7

    def __init__(self, piece_index: int, begin: int, block: bytes):
        self.piece_index = piece_index
        self.begin = begin
        self.block = block

    def encode(self) -> bytes:
        length = 9 + len(self.block)
        return struct.pack(f'>IBII{len(self.block)}s', length, self.MESSAGE_ID, self.piece_index, self.begin, self.block)

    @classmethod
    def decode(cls, data: bytes):
        header_len = 9 # ID (1) + Index (4) + Begin (4)
        _, piece_index, begin = struct.unpack('>BII', data[:header_len])
        block = data[header_len:]
        return cls(piece_index, begin, block)

    def __eq__(self, other):
        return (isinstance(other, Piece) and
                self.piece_index == other.piece_index and
                self.begin == other.begin and
                self.block == other.block)

class Cancel(PeerMessage):
    """<len=0013><id=8><index><begin><length>"""
    MESSAGE_ID = 8

    def __init__(self, piece_index: int, begin: int, length: int = BLOCK_SIZE):
        self.piece_index = piece_index
        self.begin = begin
        self.length = length

    def encode(self) -> bytes:
        return struct.pack('>IBIII', 13, self.MESSAGE_ID, self.piece_index, self.begin, self.length)

    @classmethod
    def decode(cls, data: bytes):
        _, piece_index, begin, length = struct.unpack('>BIII', data)
        return cls(piece_index, begin, length)

    def __eq__(self, other):
        return (isinstance(other, Cancel) and
                self.piece_index == other.piece_index and
                self.begin == other.begin and
                self.length == other.length)

# Mapping of message IDs to their corresponding classes for easy decoding
MESSAGE_ID_TO_CLASS = {
    0: Choke,
    1: Unchoke,
    2: Interested,
    3: NotInterested,
    4: Have,
    5: Bitfield,
    6: Request,
    7. Piece,
    8: Cancel
}

```

### 3.3 Implementation: The `PeerStreamIterator`

A TCP connection provides a stream of bytes, not a stream of discrete messages. A single read from the socket might return a partial message, one full message, or one-and-a-half messages.14 To handle this, we will create an asynchronous iterator,

`PeerStreamIterator`. This class will read from an `asyncio.StreamReader`, maintain an internal buffer, and only yield a fully parsed message object when enough data has been received. This pattern cleanly separates the low-level network buffering logic from the high-level message processing logic.4

Python

```
# bittorrent_client/protocol.py (continued)

import asyncio

class PeerStreamIterator:
    """
    Asynchronous iterator for reading and decoding peer messages from a stream.
    """
    def __init__(self, reader: asyncio.StreamReader):
        self._reader = reader
        self._buffer = b''

    async def __aiter__(self):
        return self

    async def __anext__(self) -> PeerMessage:
        while True:
            try:
                # First, try to parse a message from the existing buffer
                message = self._parse_from_buffer()
                if message:
                    return message

                # If no full message, read more data from the stream
                chunk = await self._reader.read(4096)
                if not chunk:
                    # Stream is closed
                    raise StopAsyncIteration
                self._buffer += chunk
            except asyncio.IncompleteReadError:
                raise StopAsyncIteration

    def _parse_from_buffer(self) -> PeerMessage | None:
        """
        Attempts to parse one message from the internal buffer.
        Returns the message if successful, None otherwise.
        """
        if len(self._buffer) < 4:
            return None # Not even enough for the length prefix

        length_prefix = struct.unpack('>I', self._buffer[:4])

        if length_prefix == 0:
            # KeepAlive message
            self._buffer = self._buffer[4:]
            return KeepAlive()

        if len(self._buffer) < 4 + length_prefix:
            return None # Not enough data for the full message

        # We have a full message
        payload = self._buffer[4 : 4 + length_prefix]
        self._buffer = self._buffer[4 + length_prefix:]

        message_id = payload
        message_class = MESSAGE_ID_TO_CLASS.get(message_id)

        if not message_class:
            # Unknown message, ignore for now
            return None

        return message_class.decode(payload)


```

### 3.4 Testing the Protocol Messages

To guarantee the correctness of our message classes, we will write a `pytest` suite that performs round-trip testing. For each message type, we will create an instance, encode it to bytes, decode it back into an object, and assert that the original and decoded objects are identical. This ensures our serialization and deserialization logic is perfectly symmetrical.

Python

```
# tests/test_protocol.py

import pytest
from bittorrent_client.protocol import (
    Handshake, KeepAlive, Choke, Unchoke, Interested, NotInterested,
    Have, Bitfield, Request, Piece, Cancel
)

def test_handshake_encode_decode():
    info_hash = b'1' * 20
    peer_id = b'2' * 20
    original = Handshake(info_hash, peer_id)
    encoded = original.encode()
    decoded = Handshake.decode(encoded)
    assert original == decoded

def test_handshake_invalid():
    with pytest.raises(ValueError):
        Handshake(b'1' * 19, b'2' * 20)
    with pytest.raises(ValueError):
        Handshake.decode(b'a' * 67)

@pytest.mark.parametrize("msg_class", [
    KeepAlive, Choke, Unchoke, Interested, NotInterested
])
def test_simple_messages_encode(msg_class):
    msg = msg_class()
    encoded = msg.encode()
    # For simple messages, we can manually check the encoding
    if isinstance(msg, KeepAlive):
        assert encoded == b'\x00\x00\x00\x00'
    else:
        assert encoded == b'\x00\x00\x00\x01' + bytes()

def test_have_encode_decode():
    original = Have(piece_index=42)
    encoded = original.encode()
    decoded = Have.decode(encoded[5:]) # Pass only payload
    assert original == decoded

def test_bitfield_encode_decode():
    bitfield_data = b'\xaa\xbb\xcc'
    original = Bitfield(bitfield_data)
    encoded = original.encode()
    decoded = Bitfield.decode(encoded[5:])
    assert original == decoded

def test_request_encode_decode():
    original = Request(piece_index=1, begin=16384, length=16384)
    encoded = original.encode()
    decoded = Request.decode(encoded[5:])
    assert original == decoded

def test_piece_encode_decode():
    block_data = b'x' * 16384
    original = Piece(piece_index=1, begin=16384, block=block_data)
    encoded = original.encode()
    decoded = Piece.decode(encoded[5:])
    assert original == decoded

def test_cancel_encode_decode():
    original = Cancel(piece_index=1, begin=16384, length=16384)
    encoded = original.encode()
    decoded = Cancel.decode(encoded[5:])
    assert original == decoded

```

A critical distinction often glossed over in the official specification is the difference between "pieces" and "blocks".4 The metainfo file divides the torrent's total data into large, fixed-size

`pieces` (e.g., 256 KB, 512 KB). These pieces are the unit of integrity; each one has a corresponding SHA1 hash in the `.torrent` file. The `Bitfield` and `Have` messages operate at this level, announcing the availability of entire pieces.14

However, transferring entire pieces at once would be inefficient. Instead, pieces are further subdivided into smaller `blocks`, typically 16 KB (2^14 bytes) in size.4 The actual data transfer messages—

`Request` and `Piece`—operate on these smaller blocks. This two-tiered system is fundamental to the client's architecture. The high-level download strategy (managed by the `PieceManager`) will think in terms of pieces, while the low-level peer communication (managed by the `PeerConnection`) will operate on blocks. A failure to separate these concerns will lead to a confusing and unmanageable implementation.

## Stage 4: Managing the Download

With the protocol machinery in place, we now need a "brain" for our client. This stage involves creating the `PieceManager`, a central component responsible for orchestrating the entire download process. It will decide which pieces to request, track their progress, verify their integrity, and write them to disk.

### 4.1 Data Models: `Piece` and `Block`

To manage the download state effectively, we'll define two simple data classes, `Piece` and `Block`, reflecting the two-tiered structure of BitTorrent data. A `Piece` will represent a large chunk of the file, containing its index, its expected SHA1 hash, and the status of all its constituent blocks. A `Block` will represent the smaller 16 KB chunk that is actually transferred, tracking its own status (`missing`, `pending`, or `retrieved`) and holding the downloaded data.

Python

```
# bittorrent_client/manager.py

import math
import hashlib
from enum import Enum

# Standard block size is 2^14 bytes
BLOCK_SIZE = 16384

class BlockStatus(Enum):
    MISSING = 0
    PENDING = 1
    RETRIEVED = 2

class Block:
    """Represents a single block within a piece."""
    def __init__(self, piece_index: int, offset: int, length: int):
        self.piece_index = piece_index
        self.offset = offset
        self.length = length
        self.status = BlockStatus.MISSING
        self.data: bytes | None = None

class Piece:
    """Represents a single piece of the torrent."""
    def __init__(self, index: int, blocks: list, hash_value: bytes):
        self.index = index
        self.blocks = blocks
        self.hash_value = hash_value

    def is_complete(self) -> bool:
        """Checks if all blocks in the piece have been retrieved."""
        return all(b.status == BlockStatus.RETRIEVED for b in self.blocks)

    def get_next_missing_block(self) -> Block | None:
        """Returns the next block that is needed for this piece."""
        for block in self.blocks:
            if block.status == BlockStatus.MISSING:
                return block
        return None

    def get_data(self) -> bytes:
        """Assembles the full piece data from its blocks."""
        return b"".join(b.data for b in sorted(self.blocks, key=lambda b: b.offset))

    def verify_hash(self) -> bool:
        """Verifies the SHA1 hash of the assembled piece data."""
        piece_data = self.get_data()
        return hashlib.sha1(piece_data).digest() == self.hash_value

```

### 4.2 Implementation: The `PieceManager`

The `PieceManager` is the strategic heart of the client.4 Its responsibilities are:

1.  **Initialization:** Upon creation, it uses the `Torrent` object to calculate the number of pieces and blocks required for the entire download and instantiates all the necessary `Piece` and `Block` objects.

2.  **Peer State Tracking:** It maintains a dictionary mapping peer IDs to the set of piece indices they have available. This is updated whenever a `Bitfield` or `Have` message is received from a peer.

3.  **Request Strategy:** It implements a `next_request(peer_id)` method. This method determines the next block that should be requested from a specific peer. For our simplified client, we will implement a straightforward "in-order" strategy: iterate through the pieces sequentially, find the first one that the peer has and that we need, and return its next missing block.4 While simple, this strategy is less efficient than the "rarest first" approach recommended for optimal swarm health.6 This choice is a deliberate simplification for educational purposes, but it represents a significant trade-off between implementation complexity and performance.

4.  **Data Handling:** It provides a `block_retrieved` method that a `PeerConnection` can call when it receives a `Piece` message. This method updates the corresponding `Block`'s status and stores its data.

5.  **Integrity and Persistence:** After a block is received, it checks if the parent `Piece` is now complete. If so, it verifies the piece's SHA1 hash. If the hash matches, it writes the piece's data to the correct location in the output file.


Python

```
# bittorrent_client/manager.py (continued)

from collections import defaultdict
from bittorrent_client.torrent import Torrent

class PieceManager:
    """
    Manages the state of pieces and blocks for a torrent download.
    """
    def __init__(self, torrent: Torrent):
        self._torrent = torrent
        self._output_file = open(self._torrent.name, "wb")

        self.peers = defaultdict(set) # peer_id -> set of piece indices
        self.pending_blocks = {} # (piece_index, offset) -> Block
        self.missing_pieces: list[Piece] = self._initialize_pieces()
        self.ongoing_pieces: list[Piece] =
        self.full_pieces: list[Piece] =

    def _initialize_pieces(self) -> list[Piece]:
        """Creates all Piece and Block objects for the torrent."""
        pieces =
        num_pieces = len(self._torrent.piece_hashes)

        for i in range(num_pieces):
            piece_offset = i * self._torrent.piece_length
            blocks =

            # Calculate piece size (last piece may be smaller)
            if i == num_pieces - 1:
                piece_size = self._torrent.total_size - piece_offset
            else:
                piece_size = self._torrent.piece_length

            num_blocks = math.ceil(piece_size / BLOCK_SIZE)
            for j in range(num_blocks):
                block_offset = j * BLOCK_SIZE

                # Calculate block length (last block may be smaller)
                if j == num_blocks - 1:
                    block_length = piece_size - block_offset
                else:
                    block_length = BLOCK_SIZE

                blocks.append(Block(i, block_offset, block_length))

            pieces.append(Piece(i, blocks, self._torrent.piece_hashes[i]))
        return pieces

    @property
    def is_complete(self) -> bool:
        """True if all pieces have been downloaded and verified."""
        return not self.missing_pieces and not self.ongoing_pieces

    def add_peer(self, peer_id: bytes, bitfield: bytes):
        """Adds a peer and their available pieces (from bitfield)."""
        num_pieces = len(self._torrent.piece_hashes)
        for i in range(num_pieces):
            byte_index = i // 8
            bit_index = i % 8
            if byte_index < len(bitfield):
                if (bitfield[byte_index] >> (7 - bit_index)) & 1:
                    self.peers[peer_id].add(i)

    def update_peer(self, peer_id: bytes, piece_index: int):
        """Updates a peer's available pieces with a 'Have' message."""
        if peer_id in self.peers:
            self.peers[peer_id].add(piece_index)

    def remove_peer(self, peer_id: bytes):
        """Removes a peer from the manager."""
        if peer_id in self.peers:
            del self.peers[peer_id]

    def next_request(self, peer_id: bytes) -> Block | None:
        """
        Determines the next block to request from a given peer.
        Implements a simple in-order strategy.
        """
        if peer_id not in self.peers:
            return None

        # 1. Prioritize ongoing pieces
        for piece in self.ongoing_pieces:
            if piece.index in self.peers[peer_id]:
                block = piece.get_next_missing_block()
                if block:
                    self.pending_blocks[(block.piece_index, block.offset)] = block
                    block.status = BlockStatus.PENDING
                    return block

        # 2. Find a new piece to start
        for i, piece in enumerate(self.missing_pieces):
            if piece.index in self.peers[peer_id]:
                # Move piece from missing to ongoing
                piece = self.missing_pieces.pop(i)
                self.ongoing_pieces.append(piece)

                block = piece.get_next_missing_block()
                if block:
                    self.pending_blocks[(block.piece_index, block.offset)] = block
                    block.status = BlockStatus.PENDING
                    return block
        return None

    def block_retrieved(self, peer_id: bytes, piece_index: int, offset: int, data: bytes):
        """
        Called by a PeerConnection when a block is successfully downloaded.
        """
        # Remove from pending requests
        block = self.pending_blocks.pop((piece_index, offset), None)
        if not block:
            return # Block was not requested or already retrieved

        block.status = BlockStatus.RETRIEVED
        block.data = data

        # Find the piece this block belongs to
        piece = next((p for p in self.ongoing_pieces if p.index == piece_index), None)
        if not piece:
            return

        if piece.is_complete():
            if piece.verify_hash():
                self._write_piece_to_disk(piece)
                self.ongoing_pieces.remove(piece)
                self.full_pieces.append(piece)
                # TODO: Send 'Have' messages to all connected peers
                print(f"Piece {piece.index} verified and saved.")
            else:
                # Hash failed, reset the piece to be downloaded again
                print(f"Piece {piece.index} hash verification failed. Resetting.")
                for b in piece.blocks:
                    b.status = BlockStatus.MISSING
                    b.data = None
                self.ongoing_pieces.remove(piece)
                self.missing_pieces.append(piece)

    def _write_piece_to_disk(self, piece: Piece):
        """Writes a verified piece to the output file."""
        position = piece.index * self._torrent.piece_length
        self._output_file.seek(position)
        self._output_file.write(piece.get_data())

    def close(self):
        """Closes the output file."""
        self._output_file.close()

```

### 4.3 Testing the `PieceManager`

Testing the `PieceManager` requires isolating it from the network and filesystem. We will use `pytest` and mocks to simulate peer availability and file writes.

Python

```
# tests/test_manager.py

import pytest
from unittest.mock import MagicMock, patch, mock_open
from bittorrent_client.torrent import Torrent
from bittorrent_client.manager import PieceManager, BlockStatus

@pytest.fixture
def mock_torrent_small():
    """A small mock torrent for manager tests."""
    metainfo = {
        b'announce': b'http://tracker',
        b'info': {
            b'name': b'small.file',
            b'piece length': 32768, # 2 blocks
            b'pieces': (b'a'*20 + b'b'*20), # 2 pieces
            b'length': 49152 # 1.5 pieces
        }
    }
    return Torrent(metainfo)

@patch('builtins.open', new_callable=mock_open)
def test_piece_manager_initialization(mock_file, mock_torrent_small):
    manager = PieceManager(mock_torrent_small)
    assert len(manager.missing_pieces) == 2

    # Piece 0 should have 2 blocks
    assert len(manager.missing_pieces.blocks) == 2
    assert manager.missing_pieces.blocks.length == 16384
    assert manager.missing_pieces.blocks.length == 16384

    # Piece 1 is the last piece, should have 1 block
    assert len(manager.missing_pieces.blocks) == 1
    assert manager.missing_pieces.blocks.length == 16384

    manager.close()

@patch('builtins.open', new_callable=mock_open)
def test_next_request_strategy(mock_file, mock_torrent_small):
    manager = PieceManager(mock_torrent_small)
    peer_id = b'peer1'

    # Peer has piece 1 but not piece 0
    manager.add_peer(peer_id, b'\x40') # 01000000 -> has piece 1

    # Should request first block of piece 1
    request = manager.next_request(peer_id)
    assert request is not None
    assert request.piece_index == 1
    assert request.offset == 0
    assert request.status == BlockStatus.PENDING
    assert len(manager.missing_pieces) == 1
    assert len(manager.ongoing_pieces) == 1
    assert manager.ongoing_pieces.index == 1

    # No more requests for this peer until block is retrieved
    assert manager.next_request(peer_id) is None

    manager.close()

@patch('builtins.open', new_callable=mock_open)
def test_block_retrieval_and_verification(mock_file, mock_torrent_small):
    manager = PieceManager(mock_torrent_small)
    peer_id = b'peer1'
    manager.add_peer(peer_id, b'\x80') # 10000000 -> has piece 0

    # Request all blocks for piece 0
    block1 = manager.next_request(peer_id)
    block2 = manager.missing_pieces.get_next_missing_block()
    block2.status = BlockStatus.PENDING
    manager.pending_blocks[(block2.piece_index, block2.offset)] = block2

    # Simulate retrieving blocks
    data1 = b'x' * 16384
    data2 = b'y' * 16384
    full_piece_data = data1 + data2

    # Mock the hash to succeed
    with patch('hashlib.sha1') as mock_sha1:
        mock_sha1.return_value.digest.return_value = b'a' * 20

        manager.block_retrieved(peer_id, block1.piece_index, block1.offset, data1)
        # Piece is not yet complete
        assert len(manager.full_pieces) == 0

        manager.block_retrieved(peer_id, block2.piece_index, block2.offset, data2)
        # Piece is now complete and verified
        assert len(manager.full_pieces) == 1
        assert manager.full_pieces.index == 0
        assert len(manager.ongoing_pieces) == 0

        # Check that file was written to
        mock_file().seek.assert_called_with(0)
        mock_file().write.assert_called_with(full_piece_data)

    manager.close()

```

## Stage 5: Asynchronous Orchestration

The final implementation stage is to assemble our components—the `Tracker`, `PieceManager`, and `protocol` messages—into a cohesive, asynchronous client. We will use `asyncio` to manage concurrent peer connections and coordinate the overall download process.

### 5.1 Implementation: The `PeerConnection` Worker

The `PeerConnection` class is an asynchronous worker responsible for the entire lifecycle of a single connection to a remote peer.4 Each instance will run in its own

`asyncio` task. Its core `start` method will:

1.  Use `asyncio.open_connection` to establish a non-blocking TCP connection.

2.  Perform the peer-to-peer handshake.

3.  Send an `Interested` message to signal our intent to download.

4.  Enter an infinite loop, using our `PeerStreamIterator` to read and process incoming messages.

5.  Based on the received messages, it will update its state (e.g., `am_choked`) and interact with the `PieceManager`.

6.  Once `unchoked` by the peer, it will continuously ask the `PieceManager` for the next block to request and send the corresponding `Request` message.

7.  When a `Piece` message arrives, it will pass the downloaded block data to the `PieceManager` for processing and verification.


Python

```
# bittorrent_client/connection.py

import asyncio
from bittorrent_client.protocol import (
    Handshake, PeerMessage, KeepAlive, Choke, Unchoke, Interested,
    NotInterested, Have, Bitfield, Request, Piece, Cancel,
    PeerStreamIterator
)
from bittorrent_client.manager import PieceManager

class PeerConnection:
    """
    Manages the connection and message exchange with a single peer.
    """
    def __init__(self, peer_queue: asyncio.Queue, info_hash: bytes,
                 peer_id: bytes, piece_manager: PieceManager):
        self._peer_queue = peer_queue
        self._info_hash = info_hash
        self._my_peer_id = peer_id
        self._piece_manager = piece_manager

        self.remote_peer_id: bytes | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

        self._am_choking = True
        self._am_interested = False
        self.peer_choking = True
        self.peer_interested = False

    async def start(self):
        """The main loop for the peer connection."""
        while True:
            ip, port = await self._peer_queue.get()
            try:
                await self._connect_and_handshake(ip, port)
                await self._message_loop()
            except (ConnectionError, asyncio.TimeoutError, ValueError) as e:
                print(f"Connection to {ip}:{port} failed: {e}")
            finally:
                self._cleanup()
                self._peer_queue.task_done()

    async def _connect_and_handshake(self, ip: str, port: int):
        """Establishes connection and performs the handshake."""
        print(f"Connecting to peer {ip}:{port}")
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=10
        )

        # Send our handshake
        handshake = Handshake(self._info_hash, self._my_peer_id)
        self._writer.write(handshake.encode())
        await self._writer.drain()

        # Receive their handshake
        handshake_data = await asyncio.wait_for(self._reader.read(Handshake.LENGTH), timeout=10)
        remote_handshake = Handshake.decode(handshake_data)

        if remote_handshake.info_hash!= self._info_hash:
            raise ValueError("Handshake info_hash mismatch")

        self.remote_peer_id = remote_handshake.peer_id
        print(f"Handshake successful with peer: {self.remote_peer_id.hex()}")

    async def _message_loop(self):
        """Processes incoming messages from the peer."""
        # Send Interested message to signal our intent
        await self._send_message(Interested())
        self._am_interested = True

        iterator = PeerStreamIterator(self._reader)
        async for message in iterator:
            if isinstance(message, KeepAlive):
                pass # No action needed
            elif isinstance(message, Choke):
                self.peer_choking = True
            elif isinstance(message, Unchoke):
                self.peer_choking = False
            elif isinstance(message, Interested):
                self.peer_interested = True
            elif isinstance(message, NotInterested):
                self.peer_interested = False
            elif isinstance(message, Have):
                self._piece_manager.update_peer(self.remote_peer_id, message.piece_index)
            elif isinstance(message, Bitfield):
                self._piece_manager.add_peer(self.remote_peer_id, message.bitfield)
            elif isinstance(message, Piece):
                self._piece_manager.block_retrieved(
                    self.remote_peer_id, message.piece_index, message.begin, message.block
                )

            # After processing a message, try to request a new block
            if not self.peer_choking and self._am_interested:
                await self._request_block()

    async def _request_block(self):
        """Requests the next available block from the piece manager."""
        block = self._piece_manager.next_request(self.remote_peer_id)
        if block:
            request = Request(block.piece_index, block.offset, block.length)
            await self._send_message(request)

    async def _send_message(self, message: PeerMessage):
        """Encodes and sends a message to the peer."""
        if self._writer and not self._writer.is_closing():
            self._writer.write(message.encode())
            await self._writer.drain()

    def _cleanup(self):
        """Cleans up the connection resources."""
        if self.remote_peer_id:
            self._piece_manager.remove_peer(self.remote_peer_id)
        if self._writer:
            self._writer.close()

        self.remote_peer_id = None
        self._reader = None
        self._writer = None
        self.peer_choking = True

```

### 5.2 Implementation: The `TorrentClient` Coordinator

The `TorrentClient` is the top-level orchestrator that brings everything together.4 Its architecture demonstrates a powerful and common

`asyncio` design pattern: the producer-consumer model using an `asyncio.Queue`.

-   **Producer:** The `TorrentClient`'s main loop acts as the producer. It periodically contacts the tracker to get a list of peers. It then puts these peer addresses into the `asyncio.Queue`.

-   **Consumers:** The `PeerConnection` tasks act as consumers. They independently and asynchronously pull peer addresses from the queue whenever they are ready to start a new connection.


This elegant design decouples the tracker communication from the peer communication. The `TorrentClient` doesn't need to know about the state of individual peer connections, and the `PeerConnection` workers don't need to know where the peer addresses come from. The queue serves as a non-blocking, asynchronous buffer between them.

Python

```
# bittorrent_client/client.py

import asyncio
import time
from bittorrent_client.torrent import Torrent
from bittorrent_client.tracker import Tracker
from bittorrent_client.manager import PieceManager
from bittorrent_client.connection import PeerConnection

MAX_PEER_CONNECTIONS = 40

class TorrentClient:
    """
    The main client class that orchestrates the download process.
    """
    def __init__(self, torrent: Torrent):
        self._torrent = torrent
        self._tracker = Tracker(torrent)
        self._piece_manager = PieceManager(torrent)
        self.abort = False

    async def start(self):
        """
        Starts the torrent download process.
        """
        peer_queue = asyncio.Queue()

        # Start the peer connection workers
        workers =

        # Main control loop
        last_announce = 0
        interval = 0

        while not self._piece_manager.is_complete and not self.abort:
            if time.time() - last_announce > interval:
                try:
                    response = await self._tracker.connect(
                        first=(last_announce == 0),
                        downloaded=0, # Simplified for this client
                        uploaded=0
                    )
                    last_announce = time.time()
                    interval = response.interval

                    # Add new peers to the queue
                    for peer in response.peers:
                        peer_queue.put_nowait(peer)

                    print(f"Announced to tracker. {len(response.peers)} peers received. Next announce in {interval}s.")
                except Exception as e:
                    print(f"Tracker announce failed: {e}")
                    interval = 300 # Wait 5 minutes before retrying

            await asyncio.sleep(1)

        # Download is complete or aborted, clean up
        self.stop(workers)
        print("Download complete." if not self.abort else "Download aborted.")

    def stop(self, workers: list):
        """Stops the client and all running tasks."""
        self.abort = True
        for worker in workers:
            worker.cancel()

        self._piece_manager.close()
        asyncio.create_task(self._tracker.close())

```

### 5.3 The Main Entrypoint

Finally, a simple `main.py` script serves as the entrypoint for our application. It handles command-line arguments (the path to the `.torrent` file), instantiates the `TorrentClient`, and starts the `asyncio` event loop.

Python

```
# main.py

import asyncio
import argparse
from bittorrent_client.torrent import Torrent
from bittorrent_client.client import TorrentClient

async def main():
    parser = argparse.ArgumentParser(description="A simplified BitTorrent client.")
    parser.add_argument("torrent_file", type=str, help="Path to the.torrent file")
    args = parser.parse_args()

    try:
        torrent = Torrent.from_file(args.torrent_file)
        client = TorrentClient(torrent)
        await client.start()
    except FileNotFoundError:
        print(f"Error: Torrent file not found at '{args.torrent_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient shutting down.")

```

## Stage 6: Ensuring Robustness with Continuous Integration

A functional client is one thing; a robust and maintainable one is another. The final stage of our guide focuses on establishing a professional development workflow using a comprehensive test suite and automated continuous integration (CI).

### 6.1 The `pytest` Test Suite

Throughout this guide, we have developed unit tests for each component in isolation. This strategy is crucial for building complex software. By testing `bencoding`, `tracker`, `protocol`, and `manager` modules independently with mocked dependencies, we can verify the correctness of each part before integrating them. This modular testing approach makes debugging significantly easier, as failures can be pinpointed to a specific component.

### 6.2 GitHub Actions Workflow

To automate the execution of our test suite, we will create a GitHub Actions workflow. This workflow will automatically run our `pytest` suite every time code is pushed to the repository or a pull request is created. This ensures that no new changes can be merged that break existing functionality, maintaining a high level of code quality and confidence.16

The following YAML file, placed at `.github/workflows/ci.yml`, defines a basic CI pipeline for our client. It sets up a Python environment, installs the necessary dependencies (`pytest`, `aiohttp`), and runs the test suite.18

YAML

```
#.github/workflows/ci.yml

name: Python CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        # requirements.txt should contain:
        # pytest
        # pytest-asyncio
        # pytest-mock
        # aiohttp

    - name: Run tests with pytest
      run: |
        pytest

```

## Conclusion and Future Enhancements

This guide has walked through the construction of a simplified BitTorrent client from the ground up, leveraging the power of Python's `asyncio` for concurrent network programming. We have dissected the protocol into its constituent parts: parsing the bencoded metainfo, communicating with an HTTP tracker, implementing the stateful peer wire protocol, and managing the download strategy. The resulting architecture, centered around a `PieceManager` for strategy and `PeerConnection` workers coordinated by a `TorrentClient`, provides a solid foundation for understanding how decentralized file sharing operates at a low level.

The client, while functional for downloading a single-file torrent, is intentionally simplified. For the ambitious developer, it serves as a starting point for numerous enhancements that would be required to build a full-featured, production-grade client 4:

-   **Seeding:** The current client only leeches (downloads). The most critical next step is to implement seeding logic. This involves responding to `Request` messages from other peers by reading data from the completed file and sending `Piece` messages back. It also requires sending `Bitfield` and `Have` messages to announce the pieces our client possesses.

-   **Efficient Piece Strategy:** The simple "in-order" piece selection strategy should be replaced with a "rarest first" algorithm. This would involve polling all connected peers to determine which pieces are least common in the swarm and prioritizing them, improving download speeds and the overall health of the swarm.

-   **Multi-File Torrents:** The client currently assumes a single file. Supporting multi-file torrents requires parsing the `files` dictionary in the metainfo and carefully managing piece data that may span across multiple file boundaries.9

-   **Advanced Protocol Features:** To be a truly modern client, one would need to implement support for UDP trackers (BEP-15), which offer lower overhead than HTTP trackers 19; the Distributed Hash Table (DHT) for "trackerless" torrents (BEP-5), which further decentralizes peer discovery 20; and Peer Exchange (PEX) to discover peers directly from other peers.

-   **Resume Functionality:** A robust client should be able to resume an incomplete download. This would involve checking the existing file on disk, verifying the hashes of the pieces already present, and initializing the `PieceManager` to only request the missing ones.


By tackling these features, a developer can build upon the foundation laid here to create a truly powerful and efficient peer-to-peer appl
