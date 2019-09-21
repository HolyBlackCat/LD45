#pragma once

#include <type_traits>

#include "utils/check.h"
#include "utils/mat.h"
#include "utils/meta.h"

namespace Graphics
{
    template <typename T> struct TransformFlat
    {
        static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>, "The base type can't have CV-qualifiers.");
        static_assert(!std::is_reference_v<T>, "The base type can't be a reference.");

        TransformFlat() = delete;
        ~TransformFlat() = delete;

        using scalar_t = T;
        using vector_t = vec2<T>;
        using matrix_t = mat2<T>;

      public:
        struct GenericExpr : Meta::with_virtual_destructor<GenericExpr>
        {
            using disable_vec_mat_operators = void; // Prevent vec/math `operator*` from messing things up.

            virtual vector_t Apply(vector_t v) const = 0;

            vector_t operator*(vector_t v) const
            {
                return Apply(v);
            }
        };

      private:
        struct GenericMatrix {};

        struct IdentityMatrix : public GenericMatrix
        {
            constexpr IdentityMatrix() {}
            constexpr vector_t Apply(vector_t vec) const {return vec;}
        };
        static constexpr IdentityMatrix Combine(IdentityMatrix, IdentityMatrix) {return {};}

        struct ScaleMatrix : public GenericMatrix
        {
            vector_t scale = vector_t(1);
            constexpr ScaleMatrix(vector_t scale) : scale(scale) {}
            constexpr vector_t Apply(vector_t vec) const {return vec * scale;}
        };
        static constexpr ScaleMatrix Combine(IdentityMatrix, ScaleMatrix a) {return a;}
        static constexpr ScaleMatrix Combine(ScaleMatrix a, IdentityMatrix) {return a;}
        static constexpr ScaleMatrix Combine(ScaleMatrix a, ScaleMatrix b) {return {vector_t(a.scale * b.scale)};}

        struct Matrix : public GenericMatrix
        {
            matrix_t matrix = matrix_t();
            constexpr Matrix(matrix_t matrix) : matrix(matrix) {}
            constexpr vector_t Apply(vector_t vec) const {return matrix * vec;}
        };
        static constexpr Matrix Combine(IdentityMatrix, Matrix a) {return a;}
        static constexpr Matrix Combine(Matrix a, IdentityMatrix) {return a;}
        static constexpr Matrix Combine(ScaleMatrix a, Matrix b) {return {matrix_t(a.scale.x * b.matrix.x, a.scale.y * b.matrix.y)};}
        static constexpr Matrix Combine(Matrix b, ScaleMatrix a) {return {matrix_t(a.scale.x * b.matrix.x, a.scale.y * b.matrix.y)};}
        static constexpr Matrix Combine(Matrix a, Matrix b) {return {matrix_t(a.matrix * b.matrix)};}


        struct GenericOffset {};

        struct ZeroOffset : public GenericOffset
        {
            constexpr ZeroOffset() {}
            constexpr vector_t Apply(vector_t vec) const {return vec;}
        };
        static constexpr ZeroOffset Combine(ZeroOffset, ZeroOffset) {return {};}

        struct Offset : public GenericOffset
        {
            vector_t offset = {};
            constexpr Offset(vector_t offset) : offset(offset) {}
            constexpr vector_t Apply(vector_t vec) const {return vec + offset;}
        };
        static constexpr Offset Combine(ZeroOffset, Offset a) {return a;}
        static constexpr Offset Combine(Offset a, ZeroOffset) {return a;}
        static constexpr Offset Combine(Offset a, Offset b) {return {vector_t(a.offset + b.offset)};}

        template <typename M, CHECK(std::is_base_of_v<GenericMatrix, M>)> static constexpr ZeroOffset Combine(M, ZeroOffset) {return {};}
        template <typename M, CHECK(std::is_base_of_v<GenericMatrix, M>)> static constexpr Offset Combine(M a, Offset b) {return {a.Apply(b.offset)};}


        template <typename M, typename O, CHECK(std::is_base_of_v<GenericMatrix, M>), CHECK(std::is_base_of_v<GenericOffset, O>)>
        struct Expr : public GenericExpr
        {
            M matrix;
            O offset;

            constexpr Expr() {}
            constexpr Expr(M matrix, O offset) : matrix(matrix), offset(offset) {}

            constexpr vector_t ApplyLow(vector_t v) const
            {
                return offset.Apply(matrix.Apply(v));
            }

            vector_t Apply(vector_t v) const override
            {
                return ApplyLow(v);
            }
        };

        template <typename M, typename O> Expr(M, O) -> Expr<M, O>;

      public:
        template <typename M1, typename O1, typename M2, typename O2>
        friend auto operator*(Expr<M1, O1> a, Expr<M2, O2> b)
        {
            auto matrix = Combine(a.matrix, b.matrix);
            auto offset = Combine(a.offset, Combine(a.matrix, b.offset));
            return Expr(matrix, offset);
        }

        struct Funcs
        {
            [[nodiscard]] static constexpr auto identity()
            {
                return Expr(IdentityMatrix{}, ZeroOffset{});
            }
            [[nodiscard]] static constexpr auto translate(vector_t v)
            {
                return Expr(IdentityMatrix{}, Offset(v));
            }
            [[nodiscard]] static constexpr auto rotate(scalar_t angle)
            {
                return Expr(Matrix(matrix_t::rotate(angle)), ZeroOffset{});
            }
            [[nodiscard]] static constexpr auto scale(scalar_t x)
            {
                return scale(vector_t(x));
            }
            [[nodiscard]] static constexpr auto scale(vector_t v)
            {
                return Expr(ScaleMatrix(v), ZeroOffset{});
            }
            [[nodiscard]] static constexpr auto flip_x(bool flip = true)
            {
                return scale(vector_t(flip ? -1 : 1, 1));
            }
            [[nodiscard]] static constexpr auto flip_y(bool flip = true)
            {
                return scale(vector_t(1, flip ? -1 : 1));
            }
            [[nodiscard]] static constexpr auto matrix(matrix_t m)
            {
                return Expr(Matrix(m), ZeroOffset{});
            }
        };
    };
}
