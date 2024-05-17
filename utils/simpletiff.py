import os
import re
import math
import struct
import numbers
import requests
import imagecodecs
import numpy
import numpy as np

from PIL import Image
from io import BytesIO
from tifffile import TIFF, TiffTags, DATATYPE, SAMPLEFORMAT, TiffFileError
from struct import unpack_from
from functools import lru_cache
from typing import Any, Optional, cast
from numpy.typing import ArrayLike, DTypeLike, NDArray

# from tifffile import TiffPage
# from utils.deepzoom import DeepZoomGenerator

def empty(*args):
    pass
if False: # set it to True for debugging
    debug = print
else:
    debug = empty

def bytes2str(
    b: bytes, /, encoding: str | None = None, errors: str = 'strict'
) -> str:
    """Return Unicode string from encoded bytes."""
    if encoding is not None:
        return b.decode(encoding, errors)
    try:
        return b.decode('utf-8', errors)
    except UnicodeDecodeError:
        return b.decode('cp1252', errors)


def bytestr(s: str | bytes, /, encoding: str = 'cp1252') -> bytes:
    """Return bytes from Unicode string, else pass through."""
    return s.encode(encoding) if isinstance(s, str) else s

def stripnull(
    string: str | bytes,
    /,
    null: str | bytes | None = None,
    *,
    first: bool = True,
) -> str | bytes:
    r"""Return string truncated at first null character.

    Use to clean NULL terminated C strings.

    >>> stripnull(b'bytes\x00\x00')
    b'bytes'
    >>> stripnull(b'bytes\x00bytes\x00\x00', first=False)
    b'bytes\x00bytes'
    >>> stripnull('string\x00')
    'string'

    """
    if null is None:
        if isinstance(string, bytes):
            null = b'\x00'
        else:
            null = '\0'
    if first:
        i = string.find(null)  # type: ignore
        return string if i < 0 else string[:i]
    null = null[0]  # type: ignore
    i = len(string)
    while i:
        i -= 1
        if string[i] != null:
            break
    else:
        i = -1
    return string[: i + 1]


class FileReader:
    def __init__(self, file):
        if file.startswith("http://") or file.startswith("https://"):
            self._remote = True
        else:  
            self._remote = False
            self.filehandle = open(file, 'rb')

        self._name = file
        self.size = self._size = -1
        # self._file = file  # reference to original argument for re-opening

    # @lru_cache
    def seek_and_read(self, offset, bytecount):
        if self._remote:
            data = BytesIO(requests.get(self._name, 
                    headers = {"Range": "bytes=%d-%d" % (offset, offset + bytecount-1)},
                    timeout=5)
                    .content).read()
            if len(data) != bytecount:
                debug('requests.get() failed with wrong number of bytes: request [ %d ] bytes, received [ %d ] bytes' % (bytecount, len(data)))                
        else:
            self.filehandle.seek(offset)
            data = self.filehandle.read(bytecount)
        return data

    def close(self):
        if not self._remote:
            self.filehandle.close()

    @property
    def name(self) -> str:
        """Name of file or stream."""
        return self._name

    @property
    def extension(self) -> str:
        """File name extension of file or stream."""
        name, ext = os.path.splitext(self._name.lower())
        if ext and name.endswith('.ome'):
            ext = '.ome' + ext
        return ext


TYPE_DICT = {1:'BYTE', 2:'ASCII', 3:"SHORT", 4: "LONG", 5: 'RATIONAL',  6:'SBYTE', 7:'UNDEFINED', 8:'SSHORT', 9:'SLONG', 10: 'SRATIONAL', 11:'FLOAT', 12:'DOUBLE'}
TAG_DICT = {
    254:'NewSubfileType',
    255:'SubfileType',
    256:'ImageWidth',
    257:'ImageLength', 
    258:'BitsPerSample', 
    259:'Compression', #1: no compression; 2: CCITT Group 3 1-D Huffman RLE encoding; 32773: PackBits compression
    262:'PhotometricInterpretation', #3: Palette color
    263:'Threshholding',
    270:'ImageDescription',
    273:'StripOffsets',
    274:'Orientation',
    277:'SamplesPerPixel',
    278:'RowsPerStrip',
    279:'StripByteCounts',
    284:'PlanarConfiguration',
    320:'ColorMap',
    322:'TileWidth',
    323:'TileLength',
    324:'TileOffsets',
    325:'TileByteCounts',
    347:'JPEGTables',
    530:'YCbCrSubsampling',
    32997:'ImageDepth',
    34675:'ICC Profile',
}

if not hasattr(TIFF, 'IMAGE_COMPRESSIONS'):
    TIFF.IMAGE_COMPRESSIONS = {
        6,  # jpeg
        7,  # jpeg
        22610,  # jpegxr
        33003,  # jpeg2k
        33004,  # jpeg2k
        33005,  # jpeg2k
        33007,  # alt_jpeg
        34712,  # jpeg2k
        34892,  # jpeg
        34933,  # png
        34934,  # jpegxr ZIF
        48124,  # jetraw
        50001,  # webp
        50002,  # jpegxl
    }


## https://github.com/cgohlke/tifffile/blob/103b6a337db6a84293562bda65fdc8c382281fe2/tifffile/tifffile.py#L20756C1-L20784C37
def jpeg_decode_colorspace(
    photometric: int,
    planarconfig: int = 1,
    # extrasamples: tuple[int, ...],
    jfif: bool = False,
):
    """Return JPEG and output color space for `jpeg_decode` function."""
    colorspace = None
    outcolorspace = None
    # if extrasamples:
    #     pass
    if photometric == 6:
        # YCBCR -> RGB
        outcolorspace = 2  # RGB
    elif photometric == 2:
        # RGB -> RGB
        if not jfif:
            # found in Aperio SVS
            colorspace = 2
        outcolorspace = 2
    elif photometric == 5:
        # CMYK
        outcolorspace = 4
    elif photometric > 3:
        outcolorspace = PHOTOMETRIC(photometric).name
    if planarconfig != 1:
        outcolorspace = 1  # decode separate planes to grayscale
    return colorspace, outcolorspace


class TiffTag:
    def __init__(self, parent, offset, code, dtype,
                 count, value, valueoffset) -> None:
        """TIFF tag structure.

        TiffTag instances are not thread-safe. All attributes are read-only.

        Parameters:
            parent:
                TIFF file tag belongs to.
            offset:
                Position of tag structure in file.
            code:
                Decimal code of tag.
            dtype:
                Data type of tag value item.
            count:
                Number of items in tag value.
            valueoffset:
                Position of tag value in file.

        """
        self.parent = parent
        self.offset = int(offset)
        self.code = int(code)
        self.count = int(count)
        self._value = value
        self.valueoffset = valueoffset
        try:
            self.dtype = DATATYPE(dtype)
        except ValueError:
            self.dtype = int(dtype)

    @classmethod
    def fromfile(cls, parent, offset, header, validate=True):
        tiff = parent.tiff

        valueoffset = offset + tiff.tagsize - tiff.tagoffsetthreshold
        code, dtype = struct.unpack(tiff.tagformat1, header[:4])
        count, value = struct.unpack(tiff.tagformat2, header[4:])

        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            msg = (
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {dtype!r}'
            )
            if validate:
                raise TiffFileError(msg) from exc
            logger().error(msg)
            return cls(parent, offset, code, dtype, count, None, 0)

        valuesize = count * struct.calcsize(valueformat)
        if (
            valuesize > tiff.tagoffsetthreshold
            or code in TIFF.TAG_READERS  # TODO: only works with offsets?
        ):
            valueoffset = struct.unpack(tiff.offsetformat, value)[0]
            if validate and code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            elif (
                valueoffset < 8
                or (parent.filehandle.size != -1 and valueoffset + valuesize > parent.filehandle.size)
            ):
                msg = (
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'invalid value offset {valueoffset}'
                )
                if validate:
                    raise TiffFileError(msg)
                logger().warning(msg)
                value = None
            elif code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            else:
                value = None
        elif dtype in {1, 2, 7}:
            # BYTES, ASCII, UNDEFINED
            value = value[:valuesize]
        elif (
            tiff.is_ndpi
            and count == 1
            and dtype in {4, 9, 13}
            and value[4:] != b'\x00\x00\x00\x00'
        ):
            # NDPI IFD or LONG, for example, in StripOffsets or StripByteCounts
            value = struct.unpack('<Q', value)
        else:
            fmt = '{}{}{}'.format(
                tiff.byteorder, count * int(valueformat[0]), valueformat[1]
            )
            value = struct.unpack(fmt, value[:valuesize])

        value = TiffTag._process_value(value, code, dtype, offset)

        return cls(parent, offset, code, dtype, count, value, valueoffset)

    @staticmethod
    def _read_value(parent, offset: int, code: int, dtype: int,
                    count: int, valueoffset: int,) -> Any:
        """Read tag value from file."""
        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {dtype!r}'
            ) from exc

        fh = parent.filehandle
        tiff = parent.tiff

        valuesize = count * struct.calcsize(valueformat)
        if valueoffset < 8 or (fh.size != -1 and valueoffset + valuesize > fh.size):
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid value offset {valueoffset}'
            )
        # if valueoffset % 2:
        #     logger().warning(
        #         f'<tifffile.TiffTag {code} @{offset}> '
        #         'value does not begin on word boundary'
        #     )

        if code in TIFF.TAG_READERS:
            fh.seek(valueoffset)
            readfunc = TIFF.TAG_READERS[code]
            value = readfunc(fh, tiff.byteorder, dtype, count, tiff.offsetsize)
        elif dtype in {1, 2, 7}:
            # BYTES, ASCII, UNDEFINED
            value = fh.seek_and_read(valueoffset, valuesize)
            if len(value) != valuesize:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'could not read all values'
                )
        elif code not in TIFF.TAG_TUPLE and count > 1024:
            fh.seek(valueoffset)
            value = read_numpy(
                fh, tiff.byteorder, dtype, count, tiff.offsetsize
            )
        else:
            fmt = '{}{}{}'.format(
                tiff.byteorder, count * int(valueformat[0]), valueformat[1]
            )
            value = struct.unpack(fmt, fh.seek_and_read(valueoffset, valuesize))
        return value

    @staticmethod
    def _process_value(
        value: Any, code: int, dtype: int, offset: int, /
    ) -> Any:
        """Process tag value."""
        if (
            value is None
            or dtype == 1  # BYTE
            or dtype == 7  # UNDEFINED
            or code in TIFF.TAG_READERS
            or not isinstance(value, (bytes, str, tuple))
        ):
            return value

        if dtype == 2:
            # TIFF ASCII fields can contain multiple strings,
            #   each terminated with a NUL
            try:
                value = bytes2str(
                    stripnull(cast(bytes, value), first=False).strip()
                )
            except UnicodeDecodeError as exc:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'coercing invalid ASCII to bytes, due to {exc!r}'
                )
            return value

        if code in TIFF.TAG_ENUM:
            t = TIFF.TAG_ENUM[code]
            try:
                value = tuple(t(v) for v in value)
            except ValueError as exc:
                if code not in {259, 317}:  # ignore compression/predictor
                    logger().warning(
                        f'<tifffile.TiffTag {code} @{offset}> '
                        f'raised {exc!r}'
                    )

        if len(value) == 1 and code not in TIFF.TAG_TUPLE:
            value = value[0]

        return value

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any, /) -> None:
        self._value = value

    @property
    def dtype_name(self) -> str:
        """Name of data type of tag value."""
        try:
            return self.dtype.name  # type: ignore
        except AttributeError:
            return f'TYPE{self.dtype}'

    @property
    def name(self) -> str:
        """Name of tag from :py:attr:`_TIFF.TAGS` registry."""
        return TIFF.TAGS.get(self.code, str(self.code))

    @property
    def dataformat(self) -> str:
        """Data type as `struct.pack` format."""
        return TIFF.DATA_FORMATS[self.dtype]

    @property
    def valuebytecount(self) -> int:
        """Number of bytes of tag value in file."""
        return self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])

    def __repr__(self) -> str:
        name = '|'.join(TIFF.TAGS.getall(self.code, []))
        if name:
            name = ' ' + name
        return f'<tifffile.TiffTag {self.code}{name} @{self.offset}>'


class TiffPage:
    subfiletype: int = 0
    imagewidth: int = 0
    imagelength: int = 0
    imagedepth: int = 1
    tilewidth: int = 0
    tilelength: int = 0
    tiledepth: int = 1
    samplesperpixel: int = 1
    bitspersample: int = 1
    sampleformat: int = 1
    rowsperstrip: int = 2**32 - 1
    compression: int = 1
    planarconfig: int = 1
    fillorder: int = 1
    photometric: int = 0
    predictor: int = 1
    extrasamples: tuple[int, ...] = ()
    subsampling: tuple[int, int] | None = None
    subifds: tuple[int, ...] | None = None
    jpegtables: bytes | None = None
    jpegheader: bytes | None = None
    software: str = ''
    description: str = ''
    description1: str = ''
    nodata: int | float = 0

    def __init__(self, offset, parent, index):
        self.parent = parent
        self.shape = ()
        self.shaped = (0, 0, 0, 0, 0)
        self.dtype = self._dtype = None
        self.axes = ''
        self.tags = tags = TiffTags()
        self.dataoffsets = ()
        self.databytecounts = ()
        if isinstance(index, int):
            self._index = (index,)
        else:
            self._index = tuple(index)

        self.offset = self._offset = offset
        self._next_offset = None
        
        self.read_ifd_block(offset)

        # self.parse_data(data)
        self.decompress = self.get_decode_fn()

    def read_ifd_block(self, offset):
        tiff = self.parent.tiff
        fh = self.parent.filehandle
        tags = self.tags

        try:
            tagno: int = struct.unpack(
                tiff.tagnoformat, fh.seek_and_read(offset, tiff.tagnosize)
            )[0]
            if tagno > 4096:
                raise ValueError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            raise TiffFileError(f'corrupted tag list @{offset}') from exc

        tagoffset = offset + tiff.tagnosize  # fh.tell()
        tagsize = tagsize_ = tiff.tagsize

        data = fh.seek_and_read(offset + tiff.tagnosize, tagsize * tagno)
        self._next_offset = struct.unpack(tiff.offsetformat, fh.seek_and_read(offset + tiff.tagnosize + tagsize * tagno, 4))[0]
        if len(data) != tagsize * tagno:
            raise TiffFileError(f'corrupted IFD structure')
        if tiff.is_ndpi:
            # patch offsets/values for 64-bit NDPI file
#             tagsize = 16
#             fh.seek(8, os.SEEK_CUR)
#             ext = fh.read(4 * tagno)  # high bits
#             data = b''.join(
#                 data[i * 12 : i * 12 + 12] + ext[i * 4 : i * 4 + 4]
#                 for i in range(tagno)
#             )
            raise NotImplementedError(f'ndpi is not supported yet.')
        
        tagindex = -tagsize
        for i in range(tagno):
            tagindex += tagsize
            tagdata = data[tagindex : tagindex + tagsize]
            # try:
            # print(f"{i}: {offset + tiff.tagsize - tiff.tagoffsetthreshold}")
            tag = TiffTag.fromfile(
                self.parent, offset=tagoffset + i * tagsize_, header=tagdata
            )
#             except TiffFileError as exc:
#                 logger().error(f'<TiffTag.fromfile> raised {exc!r}')
#                 continue
            tags.add(tag)

        if not tags:
            return  # found in FIBICS

        for code, name in TIFF.TAG_ATTRIBUTES.items():
            value = tags.valueof(code)
            if value is None:
                continue
            if code in {270, 305} and not isinstance(value, str):
                # wrong string type for software or description
                continue
            setattr(self, name, value)

        value = tags.valueof(270, index=1)
        if isinstance(value, str):
            self.description1 = value

        if self.subfiletype == 0:
            value = tags.valueof(255)  # SubfileType
            if value == 2:
                self.subfiletype = 0b1  # reduced image
            elif value == 3:
                self.subfiletype = 0b10  # multi-page

        # consolidate private tags; remove them from self.tags
        # if self.is_andor:
        #     self.andor_tags
        # elif self.is_epics:
        #     self.epics_tags
        # elif self.is_ndpi:
        #     self.ndpi_tags
        # if self.is_sis and 34853 in tags:
        #     # TODO: cannot change tag.name
        #     tags[34853].name = 'OlympusSIS2'

        # dataoffsets and databytecounts
        # TileOffsets
        self.dataoffsets = tags.valueof(324)
        if self.dataoffsets is None:
            # StripOffsets
            self.dataoffsets = tags.valueof(273)
            if self.dataoffsets is None:
                # JPEGInterchangeFormat et al.
                self.dataoffsets = tags.valueof(513)
                if self.dataoffsets is None:
                    self.dataoffsets = ()
                    logger().error(f'{self!r} missing data offset tag')
        # TileByteCounts
        self.databytecounts = tags.valueof(325)
        if self.databytecounts is None:
            # StripByteCounts
            self.databytecounts = tags.valueof(279)
            if self.databytecounts is None:
                # JPEGInterchangeFormatLength et al.
                self.databytecounts = tags.valueof(514)

        if (
            self.imagewidth == 0
            and self.imagelength == 0
            and self.dataoffsets
            and self.databytecounts
        ):
            # dimensions may be missing in some RAW formats
            # read dimensions from assumed JPEG encoded segment
            try:
                infos = fh.seek_and_read(self.dataoffsets[0], min(self.databytecounts[0], 4096))
                (
                    precision,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                ) = jpeg_shape(infos)
            except Exception:
                pass
            else:
                self.imagelength = imagelength
                self.imagewidth = imagewidth
                self.samplesperpixel = samplesperpixel
                if 258 not in tags:
                    self.bitspersample = 8 if precision <= 8 else 16
                if 262 not in tags and samplesperpixel == 3:
                    self.photometric = PHOTOMETRIC.YCBCR
                if 259 not in tags:
                    self.compression = COMPRESSION.OJPEG
                if 278 not in tags:
                    self.rowsperstrip = imagelength

        elif self.compression == 6:
            # OJPEG hack. See libtiff v4.2.0 tif_dirread.c#L4082
            if 262 not in tags:
                # PhotometricInterpretation missing
                self.photometric = PHOTOMETRIC.YCBCR
            elif self.photometric == 2:
                # RGB -> YCbCr
                self.photometric = PHOTOMETRIC.YCBCR
            if 258 not in tags:
                # BitsPerSample missing
                self.bitspersample = 8
            if 277 not in tags:
                # SamplesPerPixel missing
                if self.photometric in {2, 6}:
                    self.samplesperpixel = 3
                elif self.photometric in {0, 1}:
                    self.samplesperpixel = 3

#         elif self.is_lsm or (self.index != 0 and self.parent.is_lsm):
#             # correct non standard LSM bitspersample tags
#             tags[258]._fix_lsm_bitspersample()
#             if self.compression == 1 and self.predictor != 1:
#                 # work around bug in LSM510 software
#                 self.predictor = PREDICTOR.NONE

#         elif self.is_vista or (self.index != 0 and self.parent.is_vista):
#             # ISS Vista writes wrong ImageDepth tag
#             self.imagedepth = 1

#         elif self.is_stk:
#             # read UIC1tag again now that plane count is known
#             tag = tags.get(33628)  # UIC1tag
#             assert tag is not None
#             fh.seek(tag.valueoffset)
#             uic2tag = tags.get(33629)  # UIC2tag
#             try:
#                 tag.value = read_uic1tag(
#                     fh,
#                     tiff.byteorder,
#                     tag.dtype,
#                     tag.count,
#                     0,
#                     planecount=uic2tag.count if uic2tag is not None else 1,
#                 )
#             except Exception as exc:
#                 logger().warning(
#                     f'{self!r} <tifffile.read_uic1tag> raised {exc!r}'
#                 )

        tag = tags.get(50839)
        if tag is not None:
            # decode IJMetadata tag
            try:
                tag.value = imagej_metadata(
                    tag.value,
                    tags[50838].value,  # IJMetadataByteCounts
                    tiff.byteorder,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.imagej_metadata> raised {exc!r}'
                )

        # BitsPerSample
        value = tags.valueof(258)
        if value is not None:
            if self.bitspersample != 1:
                pass  # bitspersample was set by ojpeg hack
            elif tags[258].count == 1:
                self.bitspersample = int(value)
            else:
                # LSM might list more items than samplesperpixel
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.bitspersample = value
                else:
                    self.bitspersample = int(value[0])

        # SampleFormat
        value = tags.valueof(339)
        if value is not None:
            if tags[339].count == 1:
                try:
                    self.sampleformat = SAMPLEFORMAT(value)
                except ValueError:
                    self.sampleformat = int(value)
            else:
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    try:
                        self.sampleformat = SAMPLEFORMAT(value)
                    except ValueError:
                        self.sampleformat = int(value)
                else:
                    try:
                        self.sampleformat = SAMPLEFORMAT(value[0])
                    except ValueError:
                        self.sampleformat = int(value[0])

        if 322 in tags:  # TileWidth
            self.rowsperstrip = 0
        elif 257 in tags:  # ImageLength
            if 278 not in tags or tags[278].count > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)
            # self.stripsperimage = int(math.floor(
            #    float(self.imagelength + self.rowsperstrip - 1) /
            #    self.rowsperstrip))

        # determine dtype
        dtypestr = TIFF.SAMPLE_DTYPES.get(
            (self.sampleformat, self.bitspersample), None
        )
        if dtypestr is not None:
            dtype = numpy.dtype(dtypestr)
        else:
            dtype = None
        self.dtype = self._dtype = dtype

        # determine shape of data
        imagelength = self.imagelength
        imagewidth = self.imagewidth
        imagedepth = self.imagedepth
        samplesperpixel = self.samplesperpixel

        if self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:
                self.shaped = (
                    1,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (
                        imagedepth,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'ZYXS'
            else:
                self.shaped = (
                    samplesperpixel,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    1,
                )
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (
                        samplesperpixel,
                        imagedepth,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SZYX'
        else:
            self.shaped = (1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

        if not self.databytecounts:
            self.databytecounts = (
                product(self.shape) * (self.bitspersample // 8),
            )
            if self.compression != 1:
                logger().error(f'{self!r} missing ByteCounts tag')

        if imagelength and self.rowsperstrip and not self.is_lsm:
            # fix incorrect number of strip bytecounts and offsets
            maxstrips = (
                int(
                    math.floor(imagelength + self.rowsperstrip - 1)
                    / self.rowsperstrip
                )
                * self.imagedepth
            )
            if self.planarconfig == 2:
                maxstrips *= self.samplesperpixel
            if maxstrips != len(self.databytecounts):
                logger().error(
                    f'{self!r} incorrect StripByteCounts count '
                    f'({len(self.databytecounts)} != {maxstrips})'
                )
                self.databytecounts = self.databytecounts[:maxstrips]
            if maxstrips != len(self.dataoffsets):
                logger().error(
                    f'{self!r} incorrect StripOffsets count '
                    f'({len(self.dataoffsets)} != {maxstrips})'
                )
                self.dataoffsets = self.dataoffsets[:maxstrips]

        value = tags.valueof(42113)  # GDAL_NODATA
        if value is not None and dtype is not None:
            try:
                pytype = type(dtype.type(0).item())
                value = value.replace(',', '.')  # comma decimal separator
                self.nodata = pytype(value)
            except Exception:
                pass

#         mcustarts = tags.valueof(65426)
#         if mcustarts is not None and self.is_ndpi:
#             # use NDPI JPEG McuStarts as tile offsets
#             mcustarts = mcustarts.astype('int64')
#             high = tags.valueof(65432)
#             if high is not None:
#                 # McuStartsHighBytes
#                 high = high.astype('uint64')
#                 high <<= 32
#                 mcustarts += high.astype('int64')
#             fh.seek(self.dataoffsets[0])
#             jpegheader = fh.read(mcustarts[0])
#             try:
#                 (
#                     self.tilelength,
#                     self.tilewidth,
#                     self.jpegheader,
#                 ) = ndpi_jpeg_tile(jpegheader)
#             except ValueError as exc:
#                 logger().warning(
#                     f'{self!r} <tifffile.ndpi_jpeg_tile> raised {exc!r}'
#                 )
#             else:
#                 # TODO: optimize tuple(ndarray.tolist())
#                 databytecounts = numpy.diff(
#                     mcustarts, append=self.databytecounts[0]
#                 )
#                 self.databytecounts = tuple(databytecounts.tolist())
#                 mcustarts += self.dataoffsets[0]
#                 self.dataoffsets = tuple(mcustarts.tolist())

#     def readArray(self, val):
#         (type, count, v, o) = val
#         assert(type == 'LONG')
#         data = self.parent.filehandle.seek_and_read(o, 4 * count)
#         ret = unpack_from(self.parent.byteorder + str(count) + "I", data)
#         return ret

#     def readBytes(self, val): ## this is used for reading JPEGTables
#         (type, count, v, o) = val
#         assert(type == 'UNDEFINED')
#         ret = self.parent.filehandle.seek_and_read(o, count)
#         return ret
    
    def decode(self, data: bytes | None, index: int,
               jpegtables: bytes | None = None,
               jpegheader: bytes | None = None,
              ):
        return self.decompress(data, index, jpegtables=jpegtables, jpegheader=jpegheader)

    def get_decode_fn(self):
        if self.compression == 1:
            decompress = None
        else:
            decompress = TIFF.DECOMPRESSORS[self.compression]
        
        # normalize segments shape to [depth, length, width, contig]
        if self.is_tiled:
            stshape = (
                self.tiledepth,
                self.tilelength,
                self.tilewidth,
                self.samplesperpixel if self.planarconfig == 1 else 1,
            )
        else:
            stshape = (
                1,
                self.rowsperstrip,
                self.imagewidth,
                self.samplesperpixel if self.planarconfig == 1 else 1,
            )

        stdepth, stlength, stwidth, samples = stshape
        _, imdepth, imlength, imwidth, samples = self.shaped

        if self.is_tiled:
            width = (imwidth + stwidth - 1) // stwidth
            length = (imlength + stlength - 1) // stlength
            depth = (imdepth + stdepth - 1) // stdepth

            def indices(
                segmentindex: int, /
            ) -> tuple[
                tuple[int, int, int, int, int], tuple[int, int, int, int]
            ]:
                # return indices and shape of tile in image array
                return (
                    (
                        segmentindex // (width * length * depth),
                        (segmentindex // (width * length)) % depth * stdepth,
                        (segmentindex // width) % length * stlength,
                        segmentindex % width * stwidth,
                        0,
                    ),
                    stshape,
                )

            def reshape(
                data: NDArray[Any],
                indices: tuple[int, int, int, int, int],
                shape: tuple[int, int, int, int],
                /,
            ) -> NDArray[Any]:
                # return reshaped tile or raise TiffFileError
                size = shape[0] * shape[1] * shape[2] * shape[3]
                if data.ndim == 1 and data.size > size:
                    # decompression / unpacking might return too many bytes
                    data = data[:size]
                if data.size == size:
                    # complete tile
                    # data might be non-contiguous; cannot reshape inplace
                    return data.reshape(shape)
                try:
                    # data fills remaining space
                    # found in JPEG/PNG compressed tiles
                    return data.reshape(
                        (
                            min(imdepth - indices[1], shape[0]),
                            min(imlength - indices[2], shape[1]),
                            min(imwidth - indices[3], shape[2]),
                            samples,
                        )
                    )
                except ValueError:
                    pass
                try:
                    # data fills remaining horizontal space
                    # found in tiled GeoTIFF
                    return data.reshape(
                        (
                            min(imdepth - indices[1], shape[0]),
                            min(imlength - indices[2], shape[1]),
                            shape[2],
                            samples,
                        )
                    )
                except ValueError:
                    pass
                raise TiffFileError(
                    f'corrupted tile @ {indices} cannot be reshaped from '
                    f'{data.shape} to {shape}'
                )

            def pad(
                data: NDArray[Any], shape: tuple[int, int, int, int], /
            ) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
                # pad tile to shape
                if data.shape == shape:
                    return data, shape
                padwidth = [(0, i - j) for i, j in zip(shape, data.shape)]
                data = numpy.pad(data, padwidth, constant_values=self.nodata)
                return data, shape

            def pad_none(
                shape: tuple[int, int, int, int], /
            ) -> tuple[int, int, int, int]:
                # return shape of tile
                return shape

        else:
            # strips
            length = (imlength + stlength - 1) // stlength

            def indices(
                segmentindex: int, /
            ) -> tuple[
                tuple[int, int, int, int, int], tuple[int, int, int, int]
            ]:
                # return indices and shape of strip in image array
                indices = (
                    segmentindex // (length * imdepth),
                    (segmentindex // length) % imdepth * stdepth,
                    segmentindex % length * stlength,
                    0,
                    0,
                )
                shape = (
                    stdepth,
                    min(stlength, imlength - indices[2]),
                    stwidth,
                    samples,
                )
                return indices, shape

            def reshape(
                data: NDArray[Any],
                indices: tuple[int, int, int, int, int],
                shape: tuple[int, int, int, int],
                /,
            ) -> NDArray[Any]:
                # return reshaped strip or raise TiffFileError
                size = shape[0] * shape[1] * shape[2] * shape[3]
                if data.ndim == 1 and data.size > size:
                    # decompression / unpacking might return too many bytes
                    data = data[:size]
                if data.size == size:
                    # expected size
                    try:
                        data.shape = shape
                    except AttributeError:
                        # incompatible shape for in-place modification
                        # decoder returned non-contiguous array
                        data = data.reshape(shape)
                    return data
                datashape = data.shape
                try:
                    # too many rows?
                    data.shape = shape[0], -1, shape[2], shape[3]
                    data = data[:, : shape[1]]
                    data.shape = shape
                    return data
                except ValueError:
                    pass
                raise TiffFileError(
                    'corrupted strip cannot be reshaped from '
                    f'{datashape} to {shape}'
                )

            def pad(
                data: NDArray[Any], shape: tuple[int, int, int, int], /
            ) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
                # pad strip length to rowsperstrip
                shape = (shape[0], stlength, shape[2], shape[3])
                if data.shape == shape:
                    return data, shape
                padwidth = [
                    (0, 0),
                    (0, stlength - data.shape[1]),
                    (0, 0),
                    (0, 0),
                ]
                data = numpy.pad(data, padwidth, constant_values=self.nodata)
                return data, shape

            def pad_none(
                shape: tuple[int, int, int, int], /
            ) -> tuple[int, int, int, int]:
                # return shape of strip
                return (shape[0], stlength, shape[2], shape[3])

        if self.compression in {6, 7, 34892, 33007}:
            colorspace, outcolorspace = jpeg_decode_colorspace(
                self.photometric,
                planarconfig=1,
            )

            def decode_jpeg(data: bytes, index: int, 
                            jpegtables: bytes | None = None,
                            jpegheader: bytes | None = None,
                           ):
                segmentindex, shape = indices(index)
                data_array = imagecodecs.jpeg_decode(
                    data,
                    bitspersample=self.bitspersample,
                    tables=jpegtables,
                    header=jpegheader,
                    colorspace=colorspace,
                    outcolorspace=outcolorspace,
                    shape=shape[1:3],
                )
                return data_array, None, None

            return decode_jpeg
        elif self.compression in {65000, 65001, 65002}:
            # EER decoder requires shape and extra args
            raise NotImplementedError(f"A suitable decoder is not specified.")
        elif self.compression == 48124:
            # Jetraw requires pre-allocated output buffer
            raise NotImplementedError(f"A suitable decoder is not specified.")
        elif self.compression in TIFF.IMAGE_COMPRESSIONS:
            def decode_image(data: bytes, index: int, 
                             jpegtables: bytes | None = None,
                             jpegheader: bytes | None = None,
                            ):
                segmentindex, shape = indices(index)
                # return decoded segment, its shape, and indices in image
                data_array: NDArray[Any]
                data_array = decompress(data)  # type: ignore
                
                return data_array, None, None

            return decode_image
        
        dtype = numpy.dtype(self.parent.byteorder + self._dtype.char)

        if self.sampleformat == 5:
            # complex integer
            if unpredict is not None:
                raise NotImplementedError(
                    'unpredicting complex integers not supported'
                )

            itype = numpy.dtype(
                f'{self.parent.byteorder}i{self.bitspersample // 16}'
            )
            ftype = numpy.dtype(
                f'{self.parent.byteorder}f{dtype.itemsize // 2}'
            )

            def unpack(data: bytes, /) -> NDArray[Any]:
                # return complex integer as numpy.complex
                return numpy.frombuffer(data, itype).astype(ftype).view(dtype)

        elif self.bitspersample in {8, 16, 32, 64, 128}:
            # regular data types

            if (self.bitspersample * stwidth * samples) % 8:
                raise ValueError('data and sample size mismatch')
            if self.predictor == 3:  # PREDICTOR.FLOATINGPOINT
                # floating-point horizontal differencing decoder needs
                # raw byte order
                dtype = numpy.dtype(self._dtype.char)

            def unpack(data: bytes, /) -> NDArray[Any]:
                # return numpy array from buffer
                try:
                    # read only numpy array
                    return numpy.frombuffer(data, dtype)
                except ValueError:
                    # for example, LZW strips may be missing EOI
                    bps = self.bitspersample // 8
                    size = (len(data) // bps) * bps
                    return numpy.frombuffer(data[:size], dtype)

        elif isinstance(self.bitspersample, tuple):
            # for example, RGB 565
            def unpack(data: bytes, /) -> NDArray[Any]:
                # return numpy array from packed integers
                return unpack_rgb(data, dtype, self.bitspersample)

        elif self.bitspersample == 24 and dtype.char == 'f':
            # float24
            if unpredict is not None:
                # floatpred_decode requires numpy.float24, which does not exist
                raise NotImplementedError('unpredicting float24 not supported')

            def unpack(data: bytes, /) -> NDArray[Any]:
                # return numpy.float32 array from float24
                return imagecodecs.float24_decode(
                    data, byteorder=self.parent.byteorder
                )

        else:
            # bilevel and packed integers
            def unpack(data: bytes, /) -> NDArray[Any]:
                # return NumPy array from packed integers
                return imagecodecs.packints_decode(
                    data, dtype, self.bitspersample, runlen=stwidth * samples
                )

        def decode_other(data: bytes, index: int, 
                         jpegtables: bytes | None = None,
                         jpegheader: bytes | None = None,
                        ):
            # return decoded segment, its shape, and indices in image
            segmentindex, shape = indices(index)
            if self.fillorder == 2:
                data = imagecodecs.bitorder_decode(data)
            if decompress is not None:
                # TODO: calculate correct size for packed integers
                size = shape[0] * shape[1] * shape[2] * shape[3]
                data = decompress(data, out=size * dtype.itemsize)
            data_array = unpack(data)  # type: ignore
            # del data
            data_array = reshape(data_array, segmentindex, shape)
            data_array = data_array.astype('=' + dtype.char, copy=False)

            return data_array, None, None
        
        return decode_other

    @property
    def is_frame(self) -> bool:
        """Object is :py:class:`TiffFrame` instance."""
        return False

    @property
    def is_virtual(self) -> bool:
        """Page does not have IFD structure in file."""
        return False

    @property
    def is_subifd(self) -> bool:
        """Page is SubIFD of another page."""
        return len(self._index) > 1

    @property
    def is_reduced(self) -> bool:
        """Page is reduced image of another image."""
        return bool(self.subfiletype & 0b1)

    @property
    def is_multipage(self) -> bool:
        """Page is part of multi-page image."""
        return bool(self.subfiletype & 0b10)

    @property
    def is_mask(self) -> bool:
        """Page is transparency mask for another image."""
        return bool(self.subfiletype & 0b100)

    @property
    def is_mrc(self) -> bool:
        """Page is part of Mixed Raster Content."""
        return bool(self.subfiletype & 0b1000)

    @property
    def is_tiled(self) -> bool:
        """Page contains tiled image."""
        return self.tilewidth > 0  # return 322 in self.tags  # TileWidth

    @property
    def is_subsampled(self) -> bool:
        """Page contains chroma subsampled image."""
        if self.subsampling is not None:
            return self.subsampling != (1, 1)
        return self.photometric == 6  # YCbCr
        # RGB JPEG usually stored as subsampled YCbCr
        # self.compression == 7
        # and self.photometric == 2
        # and self.planarconfig == 1

    @property
    def is_imagej(self) -> bool:
        """Page contains ImageJ description metadata."""
        return self.imagej_description is not None

    @property
    def is_shaped(self) -> bool:
        """Page contains Tifffile JSON metadata."""
        return self.shaped_description is not None

    @property
    def is_mdgel(self) -> bool:
        """Page contains MDFileTag tag."""
        return (
            37701 not in self.tags  # AgilentBinary
            and 33445 in self.tags  # MDFileTag
        )

    @property
    def is_agilent(self) -> bool:
        """Page contains Agilent Technologies tags."""
        # tag 270 and 285 contain color names
        return 285 in self.tags and 37701 in self.tags  # AgilentBinary

    @property
    def is_mediacy(self) -> bool:
        """Page contains Media Cybernetics Id tag."""
        tag = self.tags.get(50288)  # MC_Id
        try:
            return tag is not None and tag.value[:7] == b'MC TIFF'
        except Exception:
            return False

    @property
    def is_stk(self) -> bool:
        """Page contains UIC1Tag tag."""
        return 33628 in self.tags

    @property
    def is_lsm(self) -> bool:
        """Page contains CZ_LSMINFO tag."""
        return 34412 in self.tags

    @property
    def is_fluoview(self) -> bool:
        """Page contains FluoView MM_STAMP tag."""
        return 34362 in self.tags

    @property
    def is_nih(self) -> bool:
        """Page contains NIHImageHeader tag."""
        return 43314 in self.tags

    @property
    def is_volumetric(self) -> bool:
        """Page contains SGI ImageDepth tag with value > 1."""
        return self.imagedepth > 1

    @property
    def is_vista(self) -> bool:
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    @property
    def is_metaseries(self) -> bool:
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index != 0 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    @property
    def is_ome(self) -> bool:
        """Page contains OME-XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('OME>')

    @property
    def is_scn(self) -> bool:
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('</scn>')

    @property
    def is_svs(self) -> bool:
        """Page contains Aperio metadata."""
        return self.description[:7] == 'Aperio '

    @property
    def index(self) -> int:
        """Index of page in IFD chain."""
        return self._index[-1]
            

class ImageTiles(object):
    """ Generate image tiles with in a given region or load existing tiles.
        Always call image_tiles.load_tiles() first before access other functions.
        rois return tile parameters: [x0, y0, w, h].
        coords return padded parameters: [x0, y0, w, h] in raw image, require padding.
        pad_width return pad width to fill image with patch_size, require padding.
    """
    def __init__(self, image_size, patch_size, padding=None, box=None):
        if isinstance(image_size, numbers.Number):
            image_size = (image_size, image_size)
        self.image_size = image_size
        
        if isinstance(patch_size, numbers.Number):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        
        if isinstance(padding, numbers.Number):
            padding = (padding, padding)
        self.padding = padding
        
        w, h = self.image_size
        if box is None:
            x0, y0, x1, y1 = [0, 0, w, h]
        else:
            x0, y0 = max(box[0], 0), max(box[1], 0)
            x1, y1 = min(box[2], w), min(box[3], h)
        self.box = [x0, y0, x1, y1]
        self.shape = None
    
    def load_tiles(self, tiles=None):
        # Calculate x_t and y_t
        if tiles is not None:
            self.x_t, self.y_t = tiles[:,0], tiles[:,1]
        else:
            x0, y0, x1, y1 = self.box
            w_p, h_p = self.patch_size
            self.y_t, self.x_t = np.mgrid[y0:y1:h_p, x0:x1:w_p]
        self.shape = self.x_t.shape

        return self
    
    def rois(self):
        x0, y0, x1, y1 = self.box
        w_p, h_p = self.patch_size
        
        h_t, w_t = np.minimum(h_p, y1 - self.y_t), np.minimum(w_p, x1 - self.x_t)
        # h_t, w_t = (y1 - self.y_t).clip(max=h_p), (x1 - self.x_t).clip(max=w_p)
        return np.stack([self.x_t, self.y_t, w_t, h_t], -1)
    
    def coords(self, padding=None):
        w, h = self.image_size
        w_p, h_p = self.patch_size
        w_d, h_d = padding or self.padding
        
        # we use (0, 0) instead of (x0, y0) to pad with original image
        x_s, y_s = (self.x_t - w_d).clip(0), (self.y_t - h_d).clip(0)
        w_s, h_s = np.minimum(self.x_t + w_p + w_d, w) - x_s, np.minimum(self.y_t + h_p + h_d, h) - y_s
        
        return np.stack([x_s, y_s, w_s, h_s], axis=-1)
    
    def pad_width(self, padding=None):
        w, h = self.image_size
        w_p, h_p = self.patch_size
        w_d, h_d = padding or self.padding
        
        pad_l, pad_u = (w_d - self.x_t).clip(0), (h_d - self.y_t).clip(0)
        pad_r, pad_d = (self.x_t + w_p + w_d - w).clip(0), (self.y_t + h_p + h_d - h).clip(0)
        
        return np.stack([pad_l, pad_r, pad_u, pad_d], axis=-1)


class SimpleTiff:
    def __init__(self, file, mode=None, omexml=None, **is_flags):
        for key, value in is_flags.items():
            if key[:3] == 'is_' and key[3:] in TIFF.FILE_FLAGS:
                if value is not None:
                    setattr(self, key, bool(value))
            else:
                raise TypeError(f'unexpected keyword argument: {key}')

        if mode not in {None, 'r', 'r+', 'rb', 'r+b'}:
            raise ValueError(f'invalid mode {mode!r}')
        
        self._fh = fh = FileReader(file)
        # self._files = {fh.name: self}
        self._name = file
        self._decoders = {}
        self._parent = None
        self.pages = []

        self._omexml = None
        if omexml:
            if omexml.strip()[-4:] != 'OME>':
                raise ValueError('invalid OME-XML')
            self._omexml = omexml
            self.is_ome = True
        else:
            self.is_ome = False

        # read header
        try:
            self.header = header = self._fh.seek_and_read(0, 8)
            try:
                byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
            except KeyError as exc:
                raise TiffFileError(f'not a TIFF file {header!r}') from exc

            version = struct.unpack(byteorder + 'H', header[2:4])[0]
            if version == 43:
                # BigTiff
                offsetsize, zero = struct.unpack(byteorder + 'HH', header[4:8])
                if zero != 0 or offsetsize != 8:
                    raise TiffFileError(
                        f'invalid BigTIFF offset size {(offsetsize, zero)}'
                    )
                if byteorder == '>':
                    self.tiff = TIFF.BIG_BE
                else:
                    self.tiff = TIFF.BIG_LE
            elif version == 42:
                # Classic TIFF
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                elif is_flags.get('is_ndpi', fh.extension == '.ndpi'):
                    # NDPI uses 64 bit IFD offsets
                    if is_flags.get('is_ndpi', True):
                        self.tiff = TIFF.NDPI_LE
                    else:
                        self.tiff = TIFF.CLASSIC_LE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            elif version == 0x4E31:
                # NIFF
                if byteorder == '>':
                    raise TiffFileError('invalid NIFF file')
                logger().error(f'{self!r} NIFF format not supported')
                self.tiff = TIFF.CLASSIC_LE
            elif version in {0x55, 0x4F52, 0x5352}:
                # Panasonic or Olympus RAW
                logger().error(
                    f'{self!r} RAW format 0x{version:04X} not supported'
                )
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            else:
                raise TiffFileError(f'invalid TIFF version {version}')

            # file handle is at offset to offset to first page
            # read IFDs
            offset = struct.unpack(self.tiff.offsetformat, header[4:8])[0]
            # print(self.header, offset)

#             while offset != 0:
#                 p = TiffPage(offset, self)
#                 self.pages.append(p)
#                 offset = p._next_offset
            
            page_index = 0
            p = TiffPage(offset, self, index=page_index)
            self.pages.append(p)
            while p._next_offset != 0:
                page_index += 1
                p = TiffPage(p._next_offset, self, index=page_index)
                self.pages.append(p)
        except Exception:
            self._fh.close()
            raise

        ## Link to utils_image class Slide
        self.register_entries()

    @property
    def byteorder(self):
        """Byteorder of TIFF file."""
        return self.tiff.byteorder

    @property
    def filehandle(self):
        """File handle."""
        return self._fh

    @property
    def filename(self) -> str:
        """Name of file handle."""
        return self._name

    def close(self):
        self.filehandle.close()
    
    def register_entries(self, verbose=1):
        self.magnitude = None
        self.mpp = None
        self.description = None
        self.page_indices = []
        self.level_dims = []
        self.level_downsamples = []

        slide = self
        self.description = slide.pages[0].description

        # magnification
        # print("THIS IS DESCRIPTION", self.description)
        val = re.findall(r'((?i:mag)|(?i:magnitude))(\s)*=(\s)*(?P<mag>[\d.]+)', self.description)
        self.magnitude = float(val[0][-1]) if val else None
        if self.magnitude is None:
            print(f"Didn't find magnitude in description.")

        # mpp
        val = re.findall(r'((?i:mpp))(\s)*=(\s)*(?P<mpp>[\d.]+)', self.description)
        self.mpp = float(val[0][-1]) if val else None
        if self.mpp is None:
            print(f"Didn't find mpp in description.")

        ## level_dims consistent with open_slide: (w, h), (OriginalHeight, OriginalWidth)
        level_dims, scales, page_indices = [(slide.pages[0].shape[1], slide.pages[0].shape[0])], [1.0], [0]
        for page_idx, page in enumerate(slide.pages[1:], 1):
            if 'label' in page.description or 'macro' in page.description:
                continue
            if page.tilewidth == 0 or page.tilelength == 0:
                continue
            h, w = page.shape[0], page.shape[1]
            if round(level_dims[0][0]/w) == round(level_dims[0][1]/h):
                level_dims.append((w, h))
                scales.append(level_dims[0][0]/w)
                page_indices.append(page_idx)

        order = sorted(range(len(scales)), key=lambda x: scales[x])
        self.page_indices = [page_indices[idx] for idx in order]
        self.level_dims = [level_dims[idx] for idx in order]
        self.level_downsamples = [scales[idx] for idx in order]
        self.n_levels = len(self.level_downsamples)


    @property
    def level_dimensions(self):
        return tuple(self.level_dims)
    
    def info(self):
        return {
            'magnitude': self.magnitude,
            'mpp': self.mpp,
            'level_dims': self.level_dims,
            'description': self.description,
        }

    def get_scales(self, x):
        """ x: (w, h) image_size tuple or a page index. """
        if isinstance(x, numbers.Number):
            w, h = self.level_dims[x]
        else:
            w, h = x
        
        return (w, h), [np.array([w/_[0], h/_[1]]) for _ in self.level_dims]

    def get_resize_level(self, x=None, downsample_only=False, epsilon=1e-2):
        """ Get nearest page level index for a given image_size/factor.
            x: (w, h) tuple or a downsampled scale_factor (.
            downsample_only: only pick the 
        """
        if isinstance(x, numbers.Number):
            factor = x
        else:
            w, h = x
            factor = min(self.level_dims[0][0]/w, self.level_dims[0][1]/h)
        rel_scales = np.array([d / factor for d in self.level_downsamples])
        
        if downsample_only:
            assert factor >= 1, f"Factor={factor}, cannot be downsampled."
            return np.where(rel_scales <= 1 + epsilon)[0][-1]
        else:
            return np.abs(np.log(rel_scales)).argmin()

    def deepzoom_coords(self, patch_size, padding=0, image_size=0, box=None):
        """ Generate tile coordinates.
            patch_size: patch_size of int or (patch_width, patch_height).
            page: the page index or image_size.
        """
        (w, h), scales = self.get_scales(image_size)
        tiles = ImageTiles((w, h), patch_size=patch_size, padding=padding, box=box)
        tiles.load_tiles()
        
        return tiles.coords()

    def deepzoom_dims(self, image_size=None):
        # Deep Zoom level
        z_size = image_size or self.level_dimensions[0]
        z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
            z_dimensions.append(z_size)
        
        return tuple(reversed(z_dimensions))

    def get_patch(self, x, level=0):        
        x0, y0, w, h = x
        # tifffile don't reorder page, so need convertion here. Little bit slow.
        tiff_page_idx = self.page_indices[level] % len(self.pages)
        patch = self.read_region(self.pages[tiff_page_idx], x0, y0, w, h)
        patch = Image.fromarray(patch[0])

        return patch

    # @profile
    def read_region(self, page, w0, h0, w, h, cache=None):
        """Extract a crop from a TIFF image file directory (IFD).

        Only the tiles englobing the crop area are loaded and not the whole page.
        This is usefull for large Whole slide images that can't fit int RAM.
        Parameters
        ----------
        page : TiffPage
            TIFF image file directory (IFD) from which the crop must be extracted.
        w0, h0: int (x0, y0)
            Coordinates of the top left corner of the desired crop.
        w, h: int
            Desired crop height, width.
        Returns
        -------
        out : ndarray of shape (imagedepth, h, w, sampleperpixel)
            Extracted crop.
        """
        if not page.is_tiled:
            raise ValueError("Input page must be tiled.")

        im_width = page.imagewidth
        im_height = page.imagelength
        # debug("im dimension", im_width, im_height, " and request", i0, j0, h, w, )
        if h < 1 or w < 1:
            raise ValueError(f"h={h} and w={w} must be strictly positive.")
        if h0 >= im_height:
            raise ValueError(f"h0={h0} should be smaller than im_height={im_height}.")
        if w0 >= im_width:
            raise ValueError(f"w0={w0} should be smaller than im_width={im_width}.")

        tile_width, tile_height = page.tilewidth, page.tilelength
        h1, w1 = h0 + h, w0 + w
        h0, w0 = max(0, h0), max(0, w0)
        h1, w1 = min(h0 + h, im_height), min(w0 + w, im_width)

        tile_h0, tile_w0 = h0 // tile_height, w0 // tile_width
        tile_h1, tile_w1 = np.ceil([h1 / tile_height, w1 / tile_width]).astype(int)

        tile_per_line = int(np.ceil(im_width / tile_width))

        out = np.empty((page.imagedepth,
                        (tile_h1 - tile_h0) * tile_height,
                        (tile_w1 - tile_w0) * tile_width,
                        page.samplesperpixel), dtype=page.dtype)

        fh = page.parent.filehandle

        jpegtables = page.tags.get('JPEGTables', None)
        if jpegtables is not None:
            jpegtables = jpegtables.value

        for i in range(tile_h0, tile_h1):
            for j in range(tile_w0, tile_w1):
                index = int(i * tile_per_line + j)

                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]
                
                # if index in cache:
                #     data = cache[index]
                # else:
                #     #fh.seek(offset)
                #     #data = fh.read(bytecount)
                #     data = fh.seek_and_read(offset, bytecount)
                #     cache[index] = data

                data = fh.seek_and_read(offset, bytecount)
                debug('index offset: ', index, offset, bytecount, offset + bytecount)
                tile, indices, shape = page.decode(data, index, jpegtables=jpegtables)

                im_h = (i - tile_h0) * tile_height
                im_w = (j - tile_w0) * tile_width
                out[:, im_h: im_h + tile_height, im_w: im_w + tile_width, :] = tile

        im_h0 = h0 - tile_h0 * tile_height
        im_w0 = w0 - tile_w0 * tile_width

        return out[:, im_h0: im_h0 + h, im_w0: im_w0 + w, :]

#     # @profile
#     def read_region(self, page, i0, j0, h, w, cache=None):
#         """Extract a crop from a TIFF image file directory (IFD).
        
#         Only the tiles englobing the crop area are loaded and not the whole page.
#         This is usefull for large Whole slide images that can't fit int RAM.
#         Parameters
#         ----------
#         page : TiffPage
#             TIFF image file directory (IFD) from which the crop must be extracted.
#         i0, j0: int
#             Coordinates of the top left corner of the desired crop.
#         h, w: int
#             Desired crop height, width.
#         Returns
#         -------
#         out : ndarray of shape (imagedepth, h, w, sampleperpixel)
#             Extracted crop.
#         """
#         if not page.is_tiled:
#             raise ValueError("Input page must be tiled.")

#         im_width = page.imagewidth
#         im_height = page.imagelength
#         # debug("im dimension", im_width, im_height, " and request", i0, j0, h, w, )
#         if h < 1 or w < 1:
#             raise ValueError(f"h={h} and w={w} must be strictly positive.")
#         if i0 >= im_height:
#             raise ValueError(f"i0={i0} should be smaller than im_height={im_height}.")
#         if j0 >= im_width:
#             raise ValueError(f"j0={j0} should be smaller than im_width={im_width}.")

#         tile_width, tile_height = page.tilewidth, page.tilelength
#         i1, j1 = i0 + h, j0 + w
#         i0, j0 = max(0, i0), max(0, j0)
#         i1, j1 = min(i0 + h, im_height), min(j0 + w, im_width)

#         tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
#         tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

#         tile_per_line = int(np.ceil(im_width / tile_width))

#         out = np.empty((page.imagedepth,
#                         (tile_i1 - tile_i0) * tile_height,
#                         (tile_j1 - tile_j0) * tile_width,
#                         page.samplesperpixel), dtype=page.dtype)
#         fh = page.parent.filehandle
#         jpegtables = page.tags.get('JPEGTables', None)
#         if jpegtables is not None:
#             jpegtables = jpegtables.value

#         for i in range(tile_i0, tile_i1):
#             for j in range(tile_j0, tile_j1):
#                 debug(i, j)
#                 index = int(i * tile_per_line + j)

#                 offset = page.dataoffsets[index]
#                 bytecount = page.databytecounts[index]

#                 # if index in cache:
#                 #     data = cache[index]
#                 # else:
#                 #     #fh.seek(offset)
#                 #     #data = fh.read(bytecount)
#                 #     data = fh.seek_and_read(offset, bytecount)
#                 #     cache[index] = data
#                 data = fh.seek_and_read(offset, bytecount)
#                 debug('index offset: ', index, offset, bytecount, offset + bytecount)
#                 tile, indices, shape = page.decode(data, index, jpegtables=jpegtables)

#                 im_i = (i - tile_i0) * tile_height
#                 im_j = (j - tile_j0) * tile_width
#                 out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

#         im_i0 = i0 - tile_i0 * tile_height
#         im_j0 = j0 - tile_j0 * tile_width

#         return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]


if __name__ == "__main__":
    def test():
        pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
        urlPrefix = 'http://localhost:8000/'
        localTiffFile = pathPrefix + '10021.svs'
        netTiffFile = urlPrefix + '10021.svs'

        print(SimpleTiff(localTiffFile).header)
        print(SimpleTiff(netTiffFile).header)
        assert(SimpleTiff(localTiffFile).header == SimpleTiff(netTiffFile).header) 

    import sys
    def usage():
        print("Usage:")
        print("\tpython SimpleTiff.py in.svs level col row out.jpeg")
    if len(sys.argv) != 6:
        usage()
        sys.exit(1)
    
    scriptFile, tiffFile, level, col, row, outFn = sys.argv
    level, col, row = map(int, (level, col, row))
    ret = SimpleTiff(tiffFile).get_svs_tile(level, col, row)
    fOut = open(outFn, 'wb')
    fOut.write(ret)
    fOut.close()
    print("[ %s ] level [ %d ] col [ %d ] row [ %d ] is converted into [ %s ]" % (tiffFile, level, col, row, outFn))
    sys.exit(0) 
