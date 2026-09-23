import Foundation

/// Snapshot-based undo/redo over a value-type document.
///
/// Call `record(_:)` with the state *before* a mutation; `undo`/`redo` swap the
/// current state with the neighbouring snapshot.
public struct UndoStack<State> {
    public private(set) var past: [State] = []
    public private(set) var future: [State] = []
    public let limit: Int

    public init(limit: Int = 100) {
        self.limit = max(1, limit)
    }

    public var canUndo: Bool { !past.isEmpty }
    public var canRedo: Bool { !future.isEmpty }

    public mutating func record(_ state: State) {
        past.append(state)
        if past.count > limit {
            past.removeFirst(past.count - limit)
        }
        future.removeAll()
    }

    public mutating func undo(from current: State) -> State? {
        guard let previous = past.popLast() else { return nil }
        future.append(current)
        return previous
    }

    public mutating func redo(from current: State) -> State? {
        guard let next = future.popLast() else { return nil }
        past.append(current)
        return next
    }

    public mutating func clear() {
        past.removeAll()
        future.removeAll()
    }
}
