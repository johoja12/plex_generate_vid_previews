# Movie Hub Priority Limit Design

## Goal

Limit hub-derived movie priority to the first 20 movies in each Plex hub. Movies
after position 20 remain eligible for normal queue processing but do not receive
a priority reason from that hub.

## Behavior

- Preserve Plex's returned hub order when determining the first 20 movies.
- Apply the limit independently to each individual Plex hub.
- Count only entries whose Plex media type is `movie`.
- Continue collecting non-movie hub entries under the existing behavior.
- Do not remove or override independent priority reasons. A movie after position
  20 may still become priority through another source such as On Deck.
- Preserve the existing 100-item collection ceiling for each Plex hub.

## Implementation

Update `Scheduler._collect_priority_hub_items` to track how many movie entries
have been accepted for the current hub. Skip movie entries after the twentieth
while continuing to convert eligible non-movie entries. The downstream priority
scoring and normal queue selection remain unchanged: movies omitted from the
hub-priority input receive no hub reason and therefore remain non-priority unless
another priority source applies.

## Testing

Add a scheduler regression test with more than 20 ordered movie entries and
non-movie entries. Verify that:

- the first 20 movies are collected in Plex order;
- later movies are excluded from hub-derived priority input; and
- non-movie entries are retained.

Run the focused scheduler and priority tests, followed by the full test suite.
