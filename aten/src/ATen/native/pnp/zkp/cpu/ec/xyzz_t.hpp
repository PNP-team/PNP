#pragma once

#include "jacobian_t.hpp"

namespace at {
namespace native {
template <
    class field_t,
    class field_h = typename field_t::mem_t,
    const field_h* a4 = nullptr>
class xyzz_t {
 public:
  field_t X, Y, ZZZ, ZZ;

 public:
  static const unsigned int degree = field_t::degree;

  using mem_t = xyzz_t;

  class affine_t {
    friend class xyzz_t;

   public:
    field_t X, Y;

   public:
    affine_t(const field_t& x, const field_t& y) : X(x), Y(y) {}
    inline affine_t() {}

    // inline   bool is_inf() const
    // {   return (bool)((int)X.is_zero() & (int)Y.is_zero());   }

    inline bool is_inf() const {
      return (bool)((int)X.is_zero());
    }

    inline affine_t& operator=(const xyzz_t& a) {
      Y = 1 / a.ZZZ;
      X = Y * a.ZZ; // 1/Z
      X = X ^ 2; // 1/Z^2
      X *= a.X; // X/Z^2
      Y *= a.Y; // Y/Z^3
      return *this;
    }
    inline affine_t(const xyzz_t& a) {
      *this = a;
    }

    inline operator jacobian_t<field_t>() const {
      return jacobian_t<field_t>{X, Y, field_t::one(is_inf())};
    }

    inline operator xyzz_t() const {
      xyzz_t p;
      p.X = X;
      p.Y = Y;
      p.ZZZ = p.ZZ = field_t::one(is_inf());
      return p;
    }
    using mem_t = affine_t;
  };

  class affine_inf_t {
   public:
    field_t X, Y;
    bool inf;

    inline bool is_inf() const {
      return inf;
    }

   public:
    inline operator affine_t() const {
      bool inf = is_inf();
      affine_t p;
      p.X = czero(X, inf);
      p.Y = czero(Y, inf);
      return p;
    }
    using mem_t = affine_inf_t;
  };

  template <class affine_t>
  inline xyzz_t& operator=(const affine_t& a) {
    X = a.X;
    Y = a.Y;
    ZZZ = ZZ = field_t::one(a.is_inf());
    return *this;
  }

  inline operator affine_t() const {
    return affine_t(*this);
  }

  inline operator jacobian_t<field_t>() const {
    return jacobian_t<field_t>{X * ZZ, Y * ZZZ, ZZ};
  }

  inline bool is_inf() const {
    return (bool)((int)ZZZ.is_zero() & (int)ZZ.is_zero());
  }

  inline void inf() {
    ZZZ.zero();
    ZZ.zero();
  }
  inline void cneg(bool neg) {
    ZZZ.cneg(neg);
  }

  /*
   * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
   * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
   * with twist to handle either input at infinity. Addition costs 12M+2S,
   * while conditional doubling - 4M+6M+3S.
   */
  void add(const xyzz_t& p2) {
    if (p2.is_inf()) {
      return;
    } else if (is_inf()) {
      *this = p2;
      return;
    }

    xyzz_t& p31 = *this;

    field_t U, S, P, R;

    U = p31.X * p2.ZZ; /* U1 = X1*ZZ2 */
    S = p31.Y * p2.ZZZ; /* S1 = Y1*ZZZ2 */
    P = p2.X * p31.ZZ; /* U2 = X2*ZZ1 */
    R = p2.Y * p31.ZZZ; /* S2 = Y2*ZZZ1 */
    P -= U; /* P = U2-U1 */
    R -= S; /* R = S2-S1 */

    if (!P.is_zero()) { /* X1!=X2 */
      field_t PP; /* add |p1| and |p2| */

      PP = P ^ 2; /* PP = P^2 */
#define PPP P
      PPP = P * PP; /* PPP = P*PP */
      p31.ZZ *= PP; /* ZZ3 = ZZ1*ZZ2*PP */
      p31.ZZZ *= PPP; /* ZZZ3 = ZZZ1*ZZZ2*PPP */
#define Q PP
      Q = U * PP; /* Q = U1*PP */
      p31.X = R ^ 2; /* R^2 */
      p31.X -= PPP; /* R^2-PPP */
      p31.X -= Q;
      p31.X -= Q; /* X3 = R^2-PPP-2*Q */
      Q -= p31.X;
      Q *= R; /* R*(Q-X3) */
      p31.Y = S * PPP; /* S1*PPP */
      p31.Y = Q - p31.Y; /* Y3 = R*(Q-X3)-S1*PPP */
      p31.ZZ *= p2.ZZ; /* ZZ1*ZZ2 */
      p31.ZZZ *= p2.ZZZ; /* ZZZ1*ZZZ2 */
#undef PPP
#undef Q
    } else if (R.is_zero()) { /* X1==X2 && Y1==Y2 */
      field_t M; /* double |p1| */

      U = p31.Y + p31.Y; /* U = 2*Y1 */
#define V P
#define W R
      V = U ^ 2; /* V = U^2 */
      W = U * V; /* W = U*V */
      S = p31.X * V; /* S = X1*V */
      M = p31.X ^ 2;
      M = M + M + M; /* M = 3*X1^2[+a*ZZ1^2] */
      if (a4 != nullptr) {
#ifdef __CUDA_ARCH__
        U = *a4;
        U *= p31.ZZ ^ 2;
#else
        U = p31.ZZ ^ 2;
        U *= *a4;
#endif
        M += U;
      }
      p31.X = M ^ 2;
      p31.X -= S;
      p31.X -= S; /* X3 = M^2-2*S */
      p31.Y *= W; /* W*Y1 */
      S -= p31.X;
      S *= M; /* M*(S-X3) */
      p31.Y = S - p31.Y; /* Y3 = M*(S-X3)-W*Y1 */
      p31.ZZ *= V; /* ZZ3 = V*ZZ1 */
      p31.ZZZ *= W; /* ZZZ3 = W*ZZZ1 */
#undef V
#undef W
    } else { /* X1==X2 && Y1==-Y2 */
      p31.inf(); /* set |p3| to infinity */
    }
  }

  inline void uadd(const xyzz_t& p2) {
    add(p2);
  }

  /*
   * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
   * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
   * with twists to handle even subtractions and either input at infinity.
   * Addition costs 8M+2S, while conditional doubling - 2M+4M+3S.
   */
  template <class affine_t>
  void add(const affine_t& p2, bool subtract = false) {
    xyzz_t& p31 = *this;

    if (p2.is_inf()) {
      return;
    } else if (p31.is_inf()) {
      p31 = p2;
      p31.ZZZ.cneg(subtract);
    } else {
      field_t P, R;

      R = p2.Y * p31.ZZZ; /* S2 = Y2*ZZZ1 */
      R.cneg(subtract);
      R -= p31.Y; /* R = S2-Y1 */
      P = p2.X * p31.ZZ; /* U2 = X2*ZZ1 */
      P -= p31.X; /* P = U2-X1 */

      if (!P.is_zero()) { /* X1!=X2 */
        field_t PP; /* add |p2| to |p1| */

        PP = P ^ 2; /* PP = P^2 */
#define PPP P
        PPP = P * PP; /* PPP = P*PP */
        p31.ZZ *= PP; /* ZZ3 = ZZ1*PP */
        p31.ZZZ *= PPP; /* ZZZ3 = ZZZ1*PPP */
#define Q PP
        Q = PP * p31.X; /* Q = X1*PP */
        p31.X = R ^ 2; /* R^2 */
        p31.X -= PPP; /* R^2-PPP */
        p31.X -= Q;
        p31.X -= Q; /* X3 = R^2-PPP-2*Q */
        Q -= p31.X;
        Q *= R; /* R*(Q-X3) */
        p31.Y *= PPP; /* Y1*PPP */
        p31.Y = Q - p31.Y; /* Y3 = R*(Q-X3)-Y1*PPP */
#undef Q
#undef PPP
      } else if (R.is_zero()) { /* X1==X2 && Y1==Y2 */
        field_t M; /* double |p2| */

#define U P
        U = p2.Y + p2.Y; /* U = 2*Y1 */
        p31.ZZ = U ^ 2; /* [ZZ3 =] V = U^2 */
        p31.ZZZ = p31.ZZ * U; /* [ZZZ3 =] W = U*V */
#define S R
        S = p2.X * p31.ZZ; /* S = X1*V */
        M = p2.X ^ 2;
        M = M + M + M; /* M = 3*X1^2[+a] */
        if (a4 != nullptr) {
          M += *a4;
        }
        p31.X = M ^ 2;
        p31.X -= S;
        p31.X -= S; /* X3 = M^2-2*S */
        p31.Y = p31.ZZZ * p2.Y; /* W*Y1 */
        S -= p31.X;
        S *= M; /* M*(S-X3) */
        p31.Y = S - p31.Y; /* Y3 = M*(S-X3)-W*Y1 */
#undef S
#undef U
        p31.ZZZ.cneg(subtract);
      } else { /* X1==X2 && Y1==-Y2 */
        p31.inf(); /* set |p3| to infinity */
      }
    }
  }

  template <class affine_t>
  inline void uadd(const affine_t& p2, bool subtract = false) {
    add(p2, subtract);
  }
};
} // namespace native
} // namespace at
