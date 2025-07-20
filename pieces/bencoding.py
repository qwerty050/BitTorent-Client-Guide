import collections.abc

class Decoder:
    """
    Decodes a bencoded byte string.
    """
    def __init__(self, data: bytes):
        if not isinstance(data, bytes):
            raise TypeError('The data to decode must be bytes.')
        self._data = data
        self._index = 0

    def decode(self):
        """
        Starts the decoding process.

        Returns:
            The decoded data as a Python object (int, bytes, list, or dict).

        Raises:
            ValueError: If the bencoded data is malformed.
        """
        # TODO: Implement the main decode logic that dispatches to other
        # methods based on the current character.
        # e.g., 'i' -> _decode_int, 'l' -> _decode_list, etc.
        pass

    def _read_char(self) -> str:
        """Reads the character at the current index."""
        # TODO: Implement this helper or similar logic.
        pass

    def _decode_int(self):
        """Decodes a bencoded integer."""
        # TODO: Implement integer decoding (e.g., 'i42e').
        pass

    def _decode_str(self):
        """Decodes a bencoded byte string."""
        # TODO: Implement string decoding (e.g., '4:spam').
        pass

    def _decode_list(self):
        """Decodes a bencoded list."""
        # TODO: Implement list decoding (e.g., 'l4:spami42ee').
        pass

    def _decode_dict(self):
        """Decodes a bencoded dictionary."""
        # TODO: Implement dictionary decoding (e.g., 'd3:bar4:spam3:fooi42ee').
        pass

class Encoder:
    """
    Encodes a Python object into a bencoded byte string.
    """
    def __init__(self, data):
        self._data = data

    def encode(self) -> bytes:
        """
        Starts the encoding process.

        Returns:
            The bencoded data as a bytes object.
        """
        # TODO: Implement the main encode logic that dispatches to other
        # methods based on the type of self._data.
        pass

    def _encode_obj(self, data):
        """Recursively encodes a Python object."""
        # TODO: Implement the recursive encoding logic.
        pass
