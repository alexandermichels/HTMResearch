#!/usr/bin/env python
import os
import csv
import copy
import json

from numbers import Real
from itertools import repeat

from nupic.data.field_meta import FieldMetaInfo, FieldMetaType, FieldMetaSpecial
from nupic.data import SENTINEL_VALUE_FOR_MISSING_DATA
from nupic.data.record_stream import RecordStreamIface
from nupic.data.utils import (intOrNone, floatOrNone, parseBool, parseTimestamp,
        serializeTimestamp, serializeTimestampNoMS, escape, unescape, parseSdr,
        serializeSdr, parseStringList, stripList)

from models.ARMAModels import ARMATimeSeries

class TimeSeriesStream(RecordStreamIface):

    def __init__(self, sequence_model, bookmark=None):
        super(TimeSeriesStream, self).__init__()
        self.sequence = sequence_model
        self._fields = [FieldMetaInfo("series", "float", "")]
        self._fieldCount = len(self._fields)
        if bookmark is not None:
                self.sequence.set_theta(bookmark)

    def __len__(self):
            return len(self.sequence)

    def __str__(self):
        return str(self.sequence)

    def __repr__(self):
        return str(self.sequence)

    def close(self):
        """ Close the stream
        """
        del self.sequence


    def rewind(self):
        """Put us back at the beginning of the file again. """
        self.sequence.set_theta(0)

    def getNextRecord(self, useCache=True):
        """
        Returns next available data record from the storage. If ``useCache`` is
        ``False``, then don't read ahead and don't cache any records.

        :return: a data row (a list or tuple) if available; None, if no more records
                         in the table (End of Stream - EOS); empty sequence (list or tuple)
                         when timing out while waiting for the next record.
        """
        assert self.sequence is not None
        if not self.sequence.has_next():
                raise StopIteration
        return  [ self.sequence.get() ]


    def getNextRecordIdx(self):
        """
        :returns: (int) index of the record that will be read next from
                            :meth:`getNextRecord`
        """
        return self.sequence.get_theta()

    def appendRecord(self, record):
        """
        Saves the record in the underlying storage. Should be implemented in
        subclasses.

        :param record: (object) to store
        """
        pass

    def appendRecords(self, records, progressCB=None):
        """
        Saves multiple records in the underlying storage. Should be implemented in
        subclasses.

        :param records: (list) of objects to store
        :param progressCB: (func) called after each appension
        """
        for record in records:
            self.appendRecord(record)
            if progressCB is not None:
                progressCB()


    def getBookmark(self):
        """Returns an anchor to the current position in the data. Passing this
        anchor to the constructor makes the current position to be the first
        returned record. If record is no longer in the storage, the first available
        after it will be returned.

        :returns: anchor to current position in the data.
        """
        return self.sequence.get_theta()

    def has_next(self):
        """
        Added method, allows me to access the underlying Sequence object's has_next() method. Essentially _NUM_RECORDS.
        """
        return self.sequence.has_next()

    def recordsExistAfter(self, bookmark):
        """
        :param bookmark: (int) where to start
        :returns: True if there are records left after the    bookmark.
        """
        theta = self.sequence.get_theta()
        self.sequence.set_theta(bookmark)
        _return = self.sequence.has_next()
        self.sequence.set_theta(theta)
        return _return

    def seekFromEnd(self, numRecords):
        """
        :param numRecords: (int) number of records from the end.
        :returns: (int) a bookmark numRecords from the end of the stream.
        """
        self.sequence.set_theta(-numRecords)
        return self.sequence.get_theta()

    def getStats(self):
        """
        :returns: storage stats (like min and max values of the fields).
        """
        if self._stats == None:
                min, max = self.sequence.get(), self.sequence.get()
                for value in self.times_series:
                        if value > max:
                                max = value
                        if value < min:
                                min = value
                self._stats = {"min" : min, "max" : max}
        return self._stats

    def clearStats(self):
        """Resets stats collected so far."""
        self._stats = None

    def getError(self):
        """ Not implemented
        :returns: errors saved in the storage."""
        return None

    def setError(self, error):
        """
        Not implemented
        Saves specified error in the storage.

        :param error: Error to store.
        """
        return None


    def getFieldNames(self):
        """
        :returns: (list) field names associated with the data.
        """
        return [f.name for f in self._fields]

    def getFields(self):
        """
        :returns: a sequence of :class:`~.FieldMetaInfo`
                            ``name``/``type``/``special`` tuples for each field in the stream.
        """
        if self._fields is None:
            return None
        else:
            return copy.copy(self._fields)

    def isCompleted(self):
        """
        Not implemented

        :returns: True if all records are already in the storage or False
                            if more records is expected.
        """
        return True

    def setCompleted(self, completed):
        """
        Not implemented
        Marks the stream completed.

        :param completed: (bool) is completed?
        """
        return None

    def setTimeout(self, timeout):
        """
        Set the read timeout in seconds

        :param timeout: (int or floating point)
        """
        pass

    def flush(self):
        """ Flush the file to disk """
        pass

    def next(self):
        record = self.getNextRecord()
        if record is None:
            raise StopIteration
        return record

    def get_model_type(self):
        """This is all me"""
        return str(self.sequence)

    def in_eval_set(self):
        return self.sequence.in_eval_set()

    def in_test_set(self):
        return self.sequence.in_test_set()

    def in_train_set(self):
        return self.sequence.in_train_set()

    def len_eval_set(self):
        return (self.sequence._eval_set_end - self.sequence._eval_set)

    def set_to_eval_theta(self):
        self.sequence.set_to_eval_theta()

    def set_to_test_theta(self):
        self.sequence.set_to_test_theta()

    def set_to_train_theta(self):
        self.sequence.set_to_train_theta()

    def get_range(self):
        return self.sequence.get_range()
