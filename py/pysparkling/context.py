from pyspark.context import SparkContext
from pyspark.sql.dataframe import DataFrame
from pyspark.rdd import RDD
from pyspark.sql import SQLContext
from h2o.frame import H2OFrame
from pysparkling.utils import FrameConversions as fc
import warnings

try:
    import h2o
    from h2o.frame import H2OFrame
    has_h2o = True
except Exception:
    println("H2O package is not available!")
    has_h2o = False

def _monkey_patch_H2OFrame(hc):
    @staticmethod
    def determine_java_vec_type(vec):
        if vec.isCategorical():
            return "enum"
        elif vec.isUUID():
            return "uuid"
        elif vec.isString():
            return "string"
        elif vec.isInt():
            if vec.isTime():
                return "time"
            else:
                return "int"
        else:
            return "real"


    def get_java_h2o_frame(self):
        if hasattr(self, '_java_frame'):
            return self._java_frame
        else:
            return hc._jhc.asH2OFrame(self.frame_id)

    @staticmethod
    def from_java_h2o_frame(h2o_frame, h2o_frame_id):
        fr = H2OFrame.get_frame(h2o_frame_id.toString())
        fr._java_frame = h2o_frame
        fr._backed_by_java_obj = True
        return fr
    H2OFrame.determine_java_vec_type = determine_java_vec_type
    H2OFrame.from_java_h2o_frame = from_java_h2o_frame
    H2OFrame.get_java_h2o_frame = get_java_h2o_frame


def _is_of_simple_type(rdd):
    if not isinstance(rdd, RDD):
        raise ValueError('rdd is not of type pyspark.rdd.RDD')

    if isinstance(rdd.first(), (str, int, bool, long, float)):
        return True
    else:
        return False

def _get_first(rdd):
    if rdd.isEmpty():
        raise ValueError('rdd is empty')

    return rdd.first()


class H2OContext(object):

    def __init__(self, sparkContext):
        try:
            self._do_init(sparkContext)
            # Hack H2OFrame from h2o package
            _monkey_patch_H2OFrame(self)
        except:
            raise

    def _do_init(self, sparkContext):
        self._sc = sparkContext
        # do not instantiate sqlContext when already one exists
        self._jsqlContext = self._sc._jvm.SQLContext.getOrCreate(self._sc._jsc.sc())
        self._sqlContext = SQLContext(sparkContext,self._jsqlContext)
        self._jsc = sparkContext._jsc
        self._jvm = sparkContext._jvm
        self._gw = sparkContext._gateway

        # Imports Sparkling Water into current JVM view
        # We cannot use directly Py4j to import Sparkling Water packages
        #   java_import(sc._jvm, "org.apache.spark.h2o.*")
        # because of https://issues.apache.org/jira/browse/SPARK-5185
        # So lets load class directly via classloader
        jvm = self._jvm
        sc = self._sc
        gw = self._gw

        self._jhc = jvm.org.apache.spark.h2o.H2OContext.getOrCreate(sc._jsc)
        self._client_ip = None
        self._client_port = None

    def start(self, init_h2o_client = True, strict_version_check = False):
        """
        Start H2OContext.

        It initializes H2O services on each node in Spark cluster and creates
        H2O python client.

        Parameters
        ----------
          init_h2o_client : initialize H2O Python client (default is True)
          strict_version_check : perform strict version check of H2O Python client against H2O Rest API
        """
        self._client_ip = self._jhc.h2oLocalClientIp()
        self._client_port = self._jhc.h2oLocalClientPort()

        if (has_h2o):
            if (init_h2o_client):
                h2o.init(ip=self._client_ip, port=self._client_port, strict_version_check = strict_version_check)
            return self
        else:
            println("H2O package is not available!")
            return None

    def stop(self):
        warnings.warn("H2OContext stopping is not yet supported...")
        #self._jhc.stop(False)

    def __str__(self):
        return "H2OContext: ip={}, port={} (open UI at http://{}:{} )".format(self._client_ip, self._client_port, self._client_ip, self._client_port)

    def __repr__(self):
        self.show()
        return ""

    def show(self):
        print self

    def as_spark_frame(self, h2o_frame):
        """
        Transforms given H2OFrame to Spark DataFrame

        Parameters
        ----------
          h2o_frame : H2OFrame

        Returns
        -------
          Spark DataFrame
        """
        if isinstance(h2o_frame,H2OFrame):
            j_h2o_frame = h2o_frame.get_java_h2o_frame()
            jdf = self._jhc.asDataFrame(j_h2o_frame, self._jsqlContext)
            return DataFrame(jdf,self._sqlContext)

    def as_h2o_frame(self, dataframe, framename = None):
        """
        Transforms given Spark RDD or DataFrame to H2OFrame.

        Parameters
        ----------
          dataframe : Spark RDD or DataFrame
          framename : Optional name for resulting H2OFrame

        Returns
        -------
          H2OFrame which contains data of original input Spark data structure
        """
        if isinstance(dataframe, DataFrame):
            return fc._as_h2o_frame_from_dataframe(self, dataframe, framename)
        elif isinstance(dataframe, RDD):
            # First check if the type T in RDD[T] is one of the python "primitive" types
            # String, Boolean, Int and Double (Python Long is converted to java.lang.BigInteger)
            if _is_of_simple_type(dataframe):
                first = _get_first(dataframe)
                if isinstance(first, str):
                    return fc._as_h2o_frame_from_RDD_String(self, dataframe, framename)
                elif isinstance(first, bool):
                    return fc._as_h2o_frame_from_RDD_Bool(self, dataframe, framename)
                elif isinstance(dataframe.max(), int):
                    return fc._as_h2o_frame_from_RDD_Long(self, dataframe, framename)
                elif isinstance(first, float):
                    return fc._as_h2o_frame_from_RDD_Float(self, dataframe, framename)
                elif isinstance(dataframe.max(), long):
                    raise ValueError('Numbers in RDD Too Big')
            else:
                return fc._as_h2o_frame_from_complex_type(self, dataframe, framename)

