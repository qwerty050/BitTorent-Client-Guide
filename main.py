import asyncio
import sys

# TODO: Import the Torrent and TorrentClient classes
# from pieces.torrent import Torrent
# from pieces.client import TorrentClient

async def main():
    """
    The main function for the BitTorrent client.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_torrent_file>")
        return

    torrent_file = sys.argv[1]

    # TODO:
    # 1. Create a Torrent object from the torrent file.
    # 2. Create a TorrentClient object.
    # 3. Start the download process.

    print(f"Starting download for {torrent_file}")

    # torrent = Torrent(torrent_file)
    # client = TorrentClient(torrent)
    # await client.start()

if __name__ == '__main__':
    # On Windows, the default event loop policy can cause issues with aiohttp.
    # We set a different policy to avoid this.
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
