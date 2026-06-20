import Foundation

@MainActor
final class ControlLibraryViewModel: ObservableObject {
    @Published var items: [ControlLibraryItem] = []

    func load() async {
        items = await ControlLibraryStore.shared.list()
    }

    func delete(_ item: ControlLibraryItem) async {
        await ControlLibraryStore.shared.delete(id: item.id)
        await load()
    }
}
