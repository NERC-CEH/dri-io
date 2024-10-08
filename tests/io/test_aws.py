import unittest
from unittest.mock import MagicMock, Mock
from driutils.io.aws import S3Writer, S3Reader
import boto3
from botocore.client import BaseClient
from mypy_boto3_s3.client import S3Client
from parameterized import parameterized
from botocore.exceptions import ClientError

class TestS3Writer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.s3_client: S3Client = boto3.client("s3") #type: ignore

    def test_s3_client_type(self):
        """Returns an object if s3_client is of type `boto3.client.s3`, otherwise
        raises an error"""

        # Happy path
        writer = S3Writer(self.s3_client)

        self.assertIsInstance(writer._connection, BaseClient)

        # Bad path
        
        with self.assertRaises(TypeError):
            S3Writer("not an s3 client") #type: ignore


    @parameterized.expand([1, "body", 1.123, {"key": b"bytes"}])
    def test_error_raises_if_write_without_bytes(self, body):
        """Tests that a type error is raised if the wrong type body used"""

        writer = S3Writer(self.s3_client)
        writer._connection = MagicMock()
        with self.assertRaises(TypeError):
            writer.write("bucket", "key", body)
        
        writer._connection.put_object.assert_not_called()

    def test_write_called(self):
        """Tests that the writer can be executed"""

        body = b"Test data"

        writer = S3Writer(self.s3_client)
        writer._connection = MagicMock()
        writer.write("bucket", "key", body)

        writer._connection.put_object.assert_called_once_with(Bucket="bucket", Key="key", Body=body)
        
class TestS3Reader(unittest.TestCase):
    """Test suite for the S3 client reader"""

    @classmethod
    def setUpClass(cls) -> None:

        cls.s3_client: S3Client = boto3.client("s3") #type: ignore
        cls.bucket = "my-bucket"
        cls.key = "my-key"
    def test_error_caught_if_read_fails(self) -> None:
        """Tests that a ClientError is raised if read fails"""

        reader = S3Reader(self.s3_client)
        fake_error =  ClientError(operation_name='InvalidKeyPair.Duplicate', error_response={
            'Error': {
                'Code': 'Duplicate', 
                'Message': 'This is a custom message'
            }
        })
        reader._connection.get_object = MagicMock(side_effect=fake_error)

        with self.assertRaises((RuntimeError, ClientError)):
            reader.read(self.bucket, self.key)

    def test_get_request_made(self) -> None:
        """Test that the get request is made to s3 client"""

        reader = S3Reader(self.s3_client)
        reader._connection = MagicMock()
        

        reader.read(self.bucket, self.key)

        reader._connection.get_object.assert_called_once_with(Bucket=self.bucket, Key=self.key)
