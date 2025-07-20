import pytest
from pieces.bencoding import Decoder, Encoder

# --- Decoder Tests ---

def test_decode_string():
    assert Decoder(b'4:spam').decode() == b'spam'

def test_decode_integer():
    assert Decoder(b'i42e').decode() == 42
    assert Decoder(b'i-42e').decode() == -42
    assert Decoder(b'i0e').decode() == 0

def test_decode_list():
    assert Decoder(b'l4:spami42ee').decode() == [b'spam', 42]

def test_decode_dictionary():
    decoded = Decoder(b'd3:bar4:spam3:fooi42ee').decode()
    expected = {b'bar': b'spam', b'foo': 42}
    assert decoded == expected

def test_decode_complex():
    decoded = Decoder(b'd4:infod6:lengthi123456e4:name8:test.txte12:piece lengthi32768e6:pieces20:aaaaaaaaaaaaaaaaaaaaee').decode()
    assert decoded[b'info'][b'length'] == 123456
    assert decoded[b'info'][b'pieces'] == b'a' * 20

def test_decode_invalid_type():
    with pytest.raises(TypeError):
        Decoder('i42e')

def test_decode_malformed_integer():
    with pytest.raises(ValueError):
        Decoder(b'i42').decode()  # Missing 'e'

def test_decode_malformed_string():
    with pytest.raises(ValueError):
        Decoder(b'4-spam').decode()  # Invalid separator

# --- Encoder Tests ---

def test_encode_string():
    assert Encoder(b'spam').encode() == b'4:spam'

def test_encode_integer():
    assert Encoder(42).encode() == b'i42e'
    assert Encoder(-42).encode() == b'i-42e'

def test_encode_list():
    assert Encoder([b'spam', 42]).encode() == b'l4:spami42ee'

def test_encode_dictionary():
    # Note: Dictionary keys must be sorted for canonical bencoding
    data = {b'foo': 42, b'bar': b'spam'}
    assert Encoder(data).encode() == b'd3:bar4:spam3:fooi42ee'

def test_encode_invalid_type():
    with pytest.raises(TypeError):
        Encoder({1, 2, 3}).encode() # Sets are not supported
