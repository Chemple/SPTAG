//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 4.0.2
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class AnnIndex : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  private bool swigCMemOwnBase;

  internal AnnIndex(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwnBase = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(AnnIndex obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~AnnIndex() {
    Dispose(false);
  }

  public void Dispose() {
    Dispose(true);
    global::System.GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwnBase) {
          swigCMemOwnBase = false;
          CSHARPSPTAGPINVOKE.delete_AnnIndex(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
    }
  }

  public AnnIndex(int p_dimension) : this(CSHARPSPTAGPINVOKE.new_AnnIndex__SWIG_0(p_dimension), true) {
  }

  public AnnIndex(string p_algoType, string p_valueType, int p_dimension) : this(CSHARPSPTAGPINVOKE.new_AnnIndex__SWIG_1(p_algoType, p_valueType, p_dimension), true) {
  }

  public void SetBuildParam(string p_name, string p_value, string p_section) {
    CSHARPSPTAGPINVOKE.AnnIndex_SetBuildParam(swigCPtr, p_name, p_value, p_section);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
  }

  public void SetSearchParam(string p_name, string p_value, string p_section) {
    CSHARPSPTAGPINVOKE.AnnIndex_SetSearchParam(swigCPtr, p_name, p_value, p_section);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
  }

  public bool LoadQuantizer(string p_quantizerFile) {
    bool ret = CSHARPSPTAGPINVOKE.AnnIndex_LoadQuantizer(swigCPtr, p_quantizerFile);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public void SetQuantizerADC(bool p_adc) {
    CSHARPSPTAGPINVOKE.AnnIndex_SetQuantizerADC(swigCPtr, p_adc);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
  }

  public bool BuildSPANN(bool p_normalized) {
    bool ret = CSHARPSPTAGPINVOKE.AnnIndex_BuildSPANN(swigCPtr, p_normalized);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool BuildSPANNWithMetaData(byte[] p_meta, int p_num, bool p_withMetaIndex, bool p_normalized) {
unsafe { fixed(byte* ptrp_meta = p_meta) { CSHARPSPTAGPINVOKE.WrapperArray tempp_meta = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_meta, (ulong)p_meta.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_BuildSPANNWithMetaData(swigCPtr,  tempp_meta , p_num, p_withMetaIndex, p_normalized);
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
  }

  public bool Build(byte[] p_data, int p_num, bool p_normalized) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_Build(swigCPtr,  tempp_data , p_num, p_normalized);
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
  }

  public bool BuildWithMetaData(byte[] p_data, byte[] p_meta, int p_num, bool p_withMetaIndex, bool p_normalized) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
unsafe { fixed(byte* ptrp_meta = p_meta) { CSHARPSPTAGPINVOKE.WrapperArray tempp_meta = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_meta, (ulong)p_meta.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_BuildWithMetaData(swigCPtr,  tempp_data ,  tempp_meta , p_num, p_withMetaIndex, p_normalized);
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
} }
  }

  public BasicResult[] Search(byte[] p_data, int p_resultNum) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
    {
      CSHARPSPTAGPINVOKE.WrapperArray data = CSHARPSPTAGPINVOKE.AnnIndex_Search(swigCPtr,  tempp_data , p_resultNum);
      BasicResult[] ret = new BasicResult[data._size];
      System.IntPtr ptr = data._data;
      for (ulong i = 0; i < data._size; i++) {
          CSHARPSPTAGPINVOKE.WrapperArray arr = (CSHARPSPTAGPINVOKE.WrapperArray)System.Runtime.InteropServices.Marshal.PtrToStructure(ptr, typeof(CSHARPSPTAGPINVOKE.WrapperArray));
          ret[i] = new BasicResult(arr._data, true);
          ptr += sizeof(CSHARPSPTAGPINVOKE.WrapperArray);
      }
      CSHARPSPTAGPINVOKE.deleteArrayOfWrapperArray(data._data);
      
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
}
} }
  }

  public BasicResult[] SearchWithMetaData(byte[] p_data, int p_resultNum) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
    {
      CSHARPSPTAGPINVOKE.WrapperArray data = CSHARPSPTAGPINVOKE.AnnIndex_SearchWithMetaData(swigCPtr,  tempp_data , p_resultNum);
      BasicResult[] ret = new BasicResult[data._size];
      System.IntPtr ptr = data._data;
      for (ulong i = 0; i < data._size; i++) {
          CSHARPSPTAGPINVOKE.WrapperArray arr = (CSHARPSPTAGPINVOKE.WrapperArray)System.Runtime.InteropServices.Marshal.PtrToStructure(ptr, typeof(CSHARPSPTAGPINVOKE.WrapperArray));
          ret[i] = new BasicResult(arr._data, true);
          ptr += sizeof(CSHARPSPTAGPINVOKE.WrapperArray);
      }
      CSHARPSPTAGPINVOKE.deleteArrayOfWrapperArray(data._data);
      
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
}
} }
  }

  public BasicResult[] BatchSearch(byte[] p_data, int p_vectorNum, int p_resultNum, bool p_withMetaData) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
    {
      CSHARPSPTAGPINVOKE.WrapperArray data = CSHARPSPTAGPINVOKE.AnnIndex_BatchSearch(swigCPtr,  tempp_data , p_vectorNum, p_resultNum, p_withMetaData);
      BasicResult[] ret = new BasicResult[data._size];
      System.IntPtr ptr = data._data;
      for (ulong i = 0; i < data._size; i++) {
          CSHARPSPTAGPINVOKE.WrapperArray arr = (CSHARPSPTAGPINVOKE.WrapperArray)System.Runtime.InteropServices.Marshal.PtrToStructure(ptr, typeof(CSHARPSPTAGPINVOKE.WrapperArray));
          ret[i] = new BasicResult(arr._data, true);
          ptr += sizeof(CSHARPSPTAGPINVOKE.WrapperArray);
      }
      CSHARPSPTAGPINVOKE.deleteArrayOfWrapperArray(data._data);
      
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
}
} }
  }

  public bool ReadyToServe() {
    bool ret = CSHARPSPTAGPINVOKE.AnnIndex_ReadyToServe(swigCPtr);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public void UpdateIndex() {
    CSHARPSPTAGPINVOKE.AnnIndex_UpdateIndex(swigCPtr);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
  }

  public bool Save(string p_saveFile) {
    bool ret = CSHARPSPTAGPINVOKE.AnnIndex_Save(swigCPtr, p_saveFile);
    if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool Add(byte[] p_data, int p_num, bool p_normalized) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_Add(swigCPtr,  tempp_data , p_num, p_normalized);
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
  }

  public bool AddWithMetaData(byte[] p_data, byte[] p_meta, int p_num, bool p_withMetaIndex, bool p_normalized) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
unsafe { fixed(byte* ptrp_meta = p_meta) { CSHARPSPTAGPINVOKE.WrapperArray tempp_meta = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_meta, (ulong)p_meta.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_AddWithMetaData(swigCPtr,  tempp_data ,  tempp_meta , p_num, p_withMetaIndex, p_normalized);
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
} }
  }

  public bool Delete(byte[] p_data, int p_num) {
unsafe { fixed(byte* ptrp_data = p_data) { CSHARPSPTAGPINVOKE.WrapperArray tempp_data = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_data, (ulong)p_data.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_Delete(swigCPtr,  tempp_data , p_num);
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
  }

  public bool DeleteByMetaData(byte[] p_meta) {
unsafe { fixed(byte* ptrp_meta = p_meta) { CSHARPSPTAGPINVOKE.WrapperArray tempp_meta = new CSHARPSPTAGPINVOKE.WrapperArray( (System.IntPtr)ptrp_meta, (ulong)p_meta.LongLength );
    {
      bool ret = CSHARPSPTAGPINVOKE.AnnIndex_DeleteByMetaData(swigCPtr,  tempp_meta );
      if (CSHARPSPTAGPINVOKE.SWIGPendingException.Pending) throw CSHARPSPTAGPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    }
} }
  }

  public static AnnIndex Load(string p_loaderFile) {
    AnnIndex ret = new AnnIndex(CSHARPSPTAGPINVOKE.AnnIndex_Load(p_loaderFile), true);
    return ret;
  }

  public static AnnIndex Merge(string p_indexFilePath1, string p_indexFilePath2) {
    AnnIndex ret = new AnnIndex(CSHARPSPTAGPINVOKE.AnnIndex_Merge(p_indexFilePath1, p_indexFilePath2), true);
    return ret;
  }

}
