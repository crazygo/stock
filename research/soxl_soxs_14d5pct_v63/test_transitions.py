import unittest
from path_learning import *
from audit_paths import reference_transition


class TransitionTests(unittest.TestCase):
    def test_successor_uses_next_official_session(self):
        days=pd.to_datetime(['2026-01-15','2026-01-16','2026-01-20'])
        at=pd.Timestamp('2026-01-20 09:25:30',tz=core.ET).tz_convert('UTC')
        panel=pd.DataFrame([[.1,.2,-3.]],index=[at],columns=PRIOR)
        bank={'blocks':np.array([[0,1]]),'date_ns':days[:2].asi8}
        add_successors(bank,days,panel)
        self.assertEqual(bank['successor_at_ns'][0],at.value)
        np.testing.assert_array_equal(bank['successor_states'][0],[.1,.2,-3.])
        add_successors(bank,days,panel.iloc[:0]);self.assertTrue(np.isnan(bank['successor_states']).all())

    def test_future_successor_excluded_even_when_prices_known(self):
        at=pd.Timestamp('2026-01-20 09:25:30',tz=core.ET).tz_convert('UTC')
        bank={'blocks':np.array([[0,1]]),'date_ns':pd.to_datetime(['2026-01-15','2026-01-16']).asi8,
              'release_ns':np.array([pd.Timestamp('2026-01-16 22:00Z').value]*2),
              'quotes':np.ones((2,8,2)),'states':np.ones((2,8,4)),
              'absolute_states':np.ones((2,8)),'successor_states':np.ones((1,3)),'successor_at_ns':np.array([at.value])}
        self.assertEqual(len(source_pool(bank,at,CLOCKS[0])),0)
        bank['successor_at_ns'][0]=day_cutoff(at).value-1
        np.testing.assert_array_equal(source_pool(bank,at,CLOCKS[0]),[0])
        bank['successor_at_ns'][0]=pd.NaT.value
        self.assertEqual(len(source_pool(bank,at,CLOCKS[0])),0)

    def test_identical_transition_rows_equal_fixed_draws(self):
        at=pd.Timestamp('2026-01-20 14:25:30Z');members=np.arange(130);a=np.arange(1,131,dtype=float);a/=a.sum();b=a[::-1]
        fixed,seed=draw_blocks(members,a,b,at,CLOCKS[0],10)
        dynamic,other=draw_blocks(members,a,b,at,CLOCKS[0],10,transition=np.tile(b,(130,1)))
        self.assertEqual(seed,other);np.testing.assert_array_equal(fixed,dynamic)

    def test_transition_depends_on_previously_selected_state(self):
        at=pd.Timestamp('2026-01-20 14:25:30Z');members=np.array([8,19]);a=[1.,0.];matrix=np.array([[0.,1.],[1.,0.]])
        selected,_=draw_blocks(members,a,a,at,CLOCKS[0],8,transition=matrix)
        np.testing.assert_array_equal(selected,np.tile([8,19,8,19],(2048,1)))

    def test_matrix_independent_equations_and_cache_invalidation(self):
        x=np.random.default_rng(62).normal(size=(160,3));y=x+.5
        a,e,ready=transition_matrix(x.tobytes(),y.tobytes());b,f,want=reference_transition(x.tobytes(),y.tobytes())
        self.assertEqual(ready,want);np.testing.assert_allclose(a,b,atol=1e-13);np.testing.assert_allclose(e,f,atol=1e-10)
        self.assertFalse(a.flags.writeable)
        c,_,_=transition_matrix(x.tobytes(),(y+2).tobytes());self.assertFalse(np.array_equal(a,c))


if __name__=='__main__':unittest.main()
