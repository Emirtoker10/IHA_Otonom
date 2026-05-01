"""
state_machine.py
VTOL SAR görev durum makinesi.

Durumlar:
    SCANNING      → otonom alan taraması
    DECELERATING  → kazazede tespit edildi, yavaşlanıyor
    HOVERING      → hover modunda, sensörler okuyor
    COMPUTING     → koordinat hesaplama
    TRANSMITTING  → GCS'e veri iletimi
    RELEASING     → drone ARM + ayrılma
    RESUMING      → taramaya devam
    RTL           → iniş noktasına dönüş
    ABORT         → acil durum

Geçişler tek yönlü: SCANNING → DECELERATING → HOVERING → COMPUTING → TRANSMITTING → RELEASING → RESUMING → SCANNING
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class State(Enum):
    SCANNING     = auto()
    DECELERATING = auto()
    HOVERING     = auto()
    COMPUTING    = auto()
    TRANSMITTING = auto()
    RELEASING    = auto()
    RESUMING     = auto()
    RTL          = auto()
    ABORT        = auto()


# Geçerli geçişler tablosu
_TRANSITIONS: Dict[State, list] = {
    State.SCANNING:     [State.DECELERATING, State.RTL, State.ABORT],
    State.DECELERATING: [State.HOVERING, State.SCANNING, State.ABORT],
    State.HOVERING:     [State.COMPUTING, State.ABORT],
    State.COMPUTING:    [State.TRANSMITTING, State.HOVERING, State.ABORT],
    State.TRANSMITTING: [State.RELEASING, State.ABORT],
    State.RELEASING:    [State.RESUMING, State.ABORT],
    State.RESUMING:     [State.SCANNING, State.RTL, State.ABORT],
    State.RTL:          [],
    State.ABORT:        [],
}


class StateMachine:
    """
    Görev durum makinesi.

    Kullanım:
        sm = StateMachine()
        sm.on_enter(State.HOVERING, my_hover_callback)
        sm.transition(State.DECELERATING)
        sm.current  # → State.DECELERATING
    """

    def __init__(self, initial: State = State.SCANNING):
        self._state = initial
        self._callbacks: Dict[State, list] = {s: [] for s in State}
        self._history: list = [(initial, time.time())]
        logger.info(f"StateMachine başladı: {initial.name}")

    # ------------------------------------------------------------------
    @property
    def current(self) -> State:
        return self._state

    def transition(self, new_state: State, reason: str = "") -> bool:
        """
        Yeni duruma geç.
        Geçersiz geçişte False döner, state değişmez.
        """
        allowed = _TRANSITIONS.get(self._state, [])
        if new_state not in allowed:
            logger.error(
                f"Geçersiz geçiş: {self._state.name} → {new_state.name}"
                + (f" ({reason})" if reason else "")
            )
            return False

        old = self._state
        self._state = new_state
        self._history.append((new_state, time.time()))

        log_msg = f"Durum: {old.name} → {new_state.name}"
        if reason:
            log_msg += f" [{reason}]"
        logger.info(log_msg)

        for cb in self._callbacks.get(new_state, []):
            try:
                cb(old, new_state)
            except Exception as e:
                logger.error(f"Callback hatası ({new_state.name}): {e}")

        return True

    def on_enter(self, state: State, callback: Callable[[State, State], None]):
        """Belirli bir duruma girildiğinde çağrılacak callback kaydeder."""
        self._callbacks[state].append(callback)

    def is_in(self, *states: State) -> bool:
        return self._state in states

    def abort(self, reason: str = ""):
        """Acil durum — ABORT'a zorla geçiş."""
        self._state = State.ABORT
        self._history.append((State.ABORT, time.time()))
        logger.critical(f"ABORT! {reason}")

    @property
    def history(self) -> list:
        return list(self._history)

    def state_duration(self) -> float:
        """Mevcut durumda geçen süre (saniye)."""
        if len(self._history) < 1:
            return 0.0
        return time.time() - self._history[-1][1]
