/* -*- C -*-  (not really, but good for syntax highlighting) */
%module sparsetools

 /* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "Python.h"
#include "numpy/arrayobject.h"
#include "complex_ops.h"
#include "sparsetools.h"
%}

%feature("autodoc", "1");

%include "numpy.i"

%init %{
    import_array();
%}



 /*
  * IN types
  */
%define I_IN_ARRAY1( ctype )
%apply ctype * IN_ARRAY1 {
    const ctype Ap [ ],
    const ctype Ai [ ],
    const ctype Aj [ ],
    const ctype Bp [ ],
    const ctype Bi [ ],	
    const ctype Bj [ ],
    const ctype offsets [ ]
};
%enddef

%define T_IN_ARRAY1( ctype )
%apply ctype * IN_ARRAY1 {
    const ctype Ax [ ],
    const ctype Bx [ ],
    const ctype Xx [ ],
    const ctype Yx [ ]
};
%enddef

%define T_IN_ARRAY2( ctype )
%apply ctype * IN_ARRAY2 {
  const ctype Mx    [ ],
  const ctype diags [ ]
};
%enddef


I_IN_ARRAY1( int  )
I_IN_ARRAY1( long )

T_IN_ARRAY1( int         )
T_IN_ARRAY1( long        )
T_IN_ARRAY1( float       )
T_IN_ARRAY1( double      )
T_IN_ARRAY1( npy_cfloat_wrapper  )
T_IN_ARRAY1( npy_cdouble_wrapper )

T_IN_ARRAY2( int         )
T_IN_ARRAY2( long        )
T_IN_ARRAY2( float       )
T_IN_ARRAY2( double      )
T_IN_ARRAY2( npy_cfloat_wrapper  )
T_IN_ARRAY2( npy_cdouble_wrapper )



 /*
  * OUT types
  */
%define I_ARRAY_ARGOUT( ctype )
%apply std::vector<ctype>* array_argout {
    std::vector<ctype>* Ap,
    std::vector<ctype>* Ai,
    std::vector<ctype>* Aj,
    std::vector<ctype>* Bp,
    std::vector<ctype>* Bi,
    std::vector<ctype>* Bj,
    std::vector<ctype>* Cp,
    std::vector<ctype>* Ci,
    std::vector<ctype>* Cj
};
%enddef

%define T_ARRAY_ARGOUT( ctype )
%apply std::vector<ctype>* array_argout {
    std::vector<ctype>* Ax, 
    std::vector<ctype>* Bx,
    std::vector<ctype>* Cx, 
    std::vector<ctype>* Xx,
    std::vector<ctype>* Yx 
};
%enddef



I_ARRAY_ARGOUT( int  )
I_ARRAY_ARGOUT( long )

T_ARRAY_ARGOUT( int                 )
T_ARRAY_ARGOUT( long                )
T_ARRAY_ARGOUT( float               )
T_ARRAY_ARGOUT( double              )
T_ARRAY_ARGOUT( npy_cfloat_wrapper  )
T_ARRAY_ARGOUT( npy_cdouble_wrapper )



 /*
  * INOUT types
  */
%define T_INPLACE_ARRAY2( ctype )
%apply ctype * INPLACE_ARRAY2 {
  ctype Mx [ ]
};
%enddef

T_INPLACE_ARRAY2( int         )
T_INPLACE_ARRAY2( long        )
T_INPLACE_ARRAY2( float       )
T_INPLACE_ARRAY2( double      )
T_INPLACE_ARRAY2( npy_cfloat_wrapper  )
T_INPLACE_ARRAY2( npy_cdouble_wrapper )



%define I_INPLACE_ARRAY1( ctype )
%apply ctype * INPLACE_ARRAY {
  ctype Ap [ ],
  ctype Aj [ ]
};
%enddef

I_INPLACE_ARRAY1( int         )
I_INPLACE_ARRAY1( long        )


%define T_INPLACE_ARRAY1( ctype )
%apply ctype * INPLACE_ARRAY {
  ctype Ax [ ]
};
%enddef

T_INPLACE_ARRAY1( long        )
T_INPLACE_ARRAY1( float       )
T_INPLACE_ARRAY1( double      )
T_INPLACE_ARRAY1( npy_cfloat_wrapper  )
T_INPLACE_ARRAY1( npy_cdouble_wrapper )






%include "sparsetools.h"
 /*
  * Order may be important here, list float before double, scalar before complex
  * 
  * Should we permit unsigned types as array indices?  Do any functions require signedness? -- Nathan (Aug 2007)
  */

%define INSTANTIATE_ALL( f_name )		     
%template(f_name)   f_name<int,int>;
%template(f_name)   f_name<int,long>;
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
%template(f_name)   f_name<int,npy_cfloat_wrapper>;
%template(f_name)   f_name<int,npy_cdouble_wrapper>;
/* 64-bit indices would go here */
%enddef


/*
 *  diag(CSR) and diag(CSC)
 */
INSTANTIATE_ALL(extract_csr_diagonal)
INSTANTIATE_ALL(extract_csc_diagonal)


/*
 *  CSR->CSC or CSC->CSR or CSR = CSR^T or CSC = CSC^T
 */
INSTANTIATE_ALL(csrtocsc)
INSTANTIATE_ALL(csctocsr)

/*
 * CSR<->COO and CSC<->COO
 */
INSTANTIATE_ALL(csrtocoo)
INSTANTIATE_ALL(csctocoo)
INSTANTIATE_ALL(cootocsr)
INSTANTIATE_ALL(cootocsc)


/*
 * CSR*CSR and CSC*CSC
 */
INSTANTIATE_ALL(csrmucsr)
INSTANTIATE_ALL(cscmucsc)

/*
 * CSR*x and CSC*x
 */
INSTANTIATE_ALL(csrmux)
INSTANTIATE_ALL(cscmux)

/*
 * CSR (binary op) CSR and CSC (binary op) CSC
 */
INSTANTIATE_ALL(csr_elmul_csr)
INSTANTIATE_ALL(csr_eldiv_csr)
INSTANTIATE_ALL(csr_plus_csr)
INSTANTIATE_ALL(csr_minus_csr)

INSTANTIATE_ALL(csc_elmul_csc)
INSTANTIATE_ALL(csc_eldiv_csc)
INSTANTIATE_ALL(csc_plus_csc)
INSTANTIATE_ALL(csc_minus_csc)



/*
 * spdiags->CSC
 */
INSTANTIATE_ALL(spdiags)

/*
 * CSR<->Dense
 */
INSTANTIATE_ALL(csrtodense)
INSTANTIATE_ALL(densetocsr)

/*
 * Sort CSR/CSC indices.
 */
INSTANTIATE_ALL(sort_csr_indices)
INSTANTIATE_ALL(sort_csc_indices)


/*
 * Sum duplicate CSR/CSC entries.
 */
INSTANTIATE_ALL(sum_csr_duplicates)
INSTANTIATE_ALL(sum_csc_duplicates)

INSTANTIATE_ALL(get_csr_submatrix)