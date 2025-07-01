import itertools
from collections.abc import Hashable, Iterator, KeysView, MutableMapping, Sequence


class _Poset[T: Hashable](MutableMapping):
    r"""Graded partially ordered set.

    Args:
        ordering (Sequence[Sequence[T]]): partial ordering of elements, where inner
            sequences contain non-comparable items.

    Raises:
        RuntimeError: `ordering` cannot contain duplicate items.

    Important:
        Type ``T`` must be a subtype of :py:type:`~typing.Hashable`.

    Tip:
        The ``rank`` is assigned automatically based on the index of the containing
        sequence in ``ordering``, so ``[["a", "b"], [], ["c"]]`` assigns ``"a"`` and
        ``"b"`` to rank 0, and ``"c"`` to rank 2.

    Note:
        This class duplicates the underlying storage into a "flat" and a "tall"
        representation, prioritizing speed over space efficiency.
    """

    _elems: list[dict[T, int]]
    _index: dict[T, int]

    def __init__(self, ordering: Sequence[Sequence[T]]) -> None:
        self._elems = []
        self._index = {}

        for rank, items in enumerate(ordering):
            if any(k in self._index for k in items):
                raise RuntimeError(
                    "`ordering` contains duplicate entries with different priorities"
                )
            rankitems = {e: rank for e in items}
            self._index |= rankitems
            self._elems += [rankitems]

    def __contains__(self, item: object) -> bool:
        return item in self._index

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Poset):
            return self._index == other._index
        else:
            return False

    def __getitem__(self, item: T) -> int:
        try:
            return self._index[item]
        except KeyError:
            raise KeyError(f"poset contains no element '{item}'")

    def __setitem__(self, item: T, rank: int) -> None:
        # check that rank is an integer
        if not isinstance(rank, int):
            raise TypeError("`rank` must be of type `int`")

        # negative index conversion
        if rank < 0:
            if rank < -self.nranks:
                raise RuntimeError(
                    f"cannot interpret `rank` {rank} for a poset with {self.nranks} ranks"
                )
            else:
                rank = rank % self.nranks

        # expand stored elements list as needed
        self._elems += [{} for _ in range(rank - self.nranks + 1)]

        # remove item from elements if required
        if item in self._index:
            del self._elems[self._index[item]][item]

        # update/insert item
        self._index[item] = rank
        self._elems[rank][item] = rank

    def __delitem__(self, item: T) -> None:
        try:
            rank = self._index[item]
        except KeyError:
            raise KeyError(f"poset contains no element '{item}'")
        else:
            del self._index[item]
            del self._elems[rank][item]

            # cleanup unused ranks
            for r in reversed(range(self.nranks)):
                if len(self._elems[r]):
                    break
                else:
                    del self._elems[r]

    def __iter__(self) -> Iterator[T]:
        return iter(itertools.chain(*(group.keys() for group in self._elems)))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def nranks(self) -> int:
        r"""Returns the number of ranks in the poset.

        Returns:
            int: number of ranks in the poset.
        """
        return len(self._elems)

    def rank(self, rank: int) -> KeysView[T]:
        r"""Returns the elements of a given rank of the poset.

        Args:
            rank (int): rank of items to retrieve.

        Returns:
            KeysView[T]: items of the specified rank.

        Raises:
            IndexError: an invalid rank was specified.
        """
        try:
            return self._elems[rank].keys()
        except IndexError:
            raise IndexError(
                f"cannot access `rank` {rank} from a poset with {self.nranks} ranks"
            )
