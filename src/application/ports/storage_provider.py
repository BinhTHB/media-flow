from abc import ABC, abstractmethod

class StorageProvider(ABC):
    @abstractmethod
    def upload(self, file_path: str) -> str:
        pass

    @abstractmethod
    def download(self, file_url: str) -> str:
        pass
