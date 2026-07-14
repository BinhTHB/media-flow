from src.application.ports.storage_provider import StorageProvider

class MockStorageProvider(StorageProvider):
    def upload(self, file_path: str) -> str:
        return f"https://mock-storage.local/{file_path}"

    def download(self, file_url: str) -> str:
        return "downloaded_temp_file.mp4"
